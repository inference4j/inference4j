/*
 * Copyright 2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.inference4j.nlp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.processing.TokenLimits;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Evaluates {@link MapReduceSummarizer} and {@link ChunkedSummarizer} on long documents against plain summarization, which truncates at the
 * model's input limit.
 *
 * <p>Opt-in, never run in CI: set {@code INFERENCE4J_EVAL=true}.
 * <pre>
 * INFERENCE4J_EVAL=true ./gradlew :inference4j-core:modelTest --tests '*LongDocumentSummarizationEvaluation'
 * </pre>
 * The report is written to {@code inference4j-core/build/reports/evaluation/long-summarization.md}.
 *
 * <p><b>Data</b>, read through the Hugging Face datasets-server and cached in
 * {@code ~/.cache/inference4j/datasets}:
 * <ul>
 *   <li>BillSum test split (CC0-1.0): US Congressional bills, which put their key content first.</li>
 *   <li>GovReport test split (CC-BY-4.0, Huang et al., 2021): GAO and CRS reports, with content spread
 *       throughout and long reference summaries.</li>
 * </ul>
 * Line wraps inside a paragraph are joined; blank lines are kept. Neither model was trained on these
 * (DistilBART on news, Flan-T5 on instructions), so only the comparison between configurations is meaningful.
 */
@EnabledIfEnvironmentVariable(named = "INFERENCE4J_EVAL", matches = "true")
class LongDocumentSummarizationEvaluation {

    private static final Path DATA_ROOT = Path.of(System.getProperty("user.home"),
            ".cache", "inference4j", "datasets");
    private static final Path REPORT = Path.of("build", "reports", "evaluation", "long-summarization.md");
    private static final int THREADS = 4;

    /** A summarization dataset on the datasets-server, and which of its documents to use. */
    record Corpus(String name, String rowsUrl, String cacheDir, String textField, int documents, int minWords,
                  int maxWords) {
    }

    private static final Corpus BILLSUM = new Corpus("BillSum (bills)",
            "https://datasets-server.huggingface.co/rows?dataset=FiscalNote%2Fbillsum&config=default&split=test"
                    + "&length=100&offset=", "billsum", "text", 15, 1500, Integer.MAX_VALUE);
    private static final Corpus GOV_REPORT = new Corpus("GovReport (government reports)",
            "https://datasets-server.huggingface.co/rows?dataset=launch%2Fgov_report&config=plain_text"
                    + "&split=test&length=100&offset=", "gov-report", "document", 10, 4000, 8000);

    record Document(String text, String reference) {
    }

    enum Mode { PLAIN, MAP_REDUCE, CHUNKED }

    /** One way of summarizing; {@code maxChunkTokens} is {@code null} for the decorator's default. */
    record Config(String name, Mode mode, Integer maxChunkTokens) {
    }

    record Model(String name, Supplier<Summarizer> build) {
    }

    /** Counts calls. */
    static final class CountingSummarizer implements Summarizer {
        private final Summarizer delegate;
        int calls;

        CountingSummarizer(Summarizer delegate) {
            this.delegate = delegate;
        }

        @Override
        public GenerationResult summarize(String text, Consumer<String> tokenListener) {
            calls++;
            return delegate.summarize(text, tokenListener);
        }

        @Override
        public TokenLimits summarizationLimits() {
            return delegate.summarizationLimits();
        }

        @Override
        public void close() throws Exception {
            delegate.close();
        }
    }

    @Test
    void evaluateOnLongDocuments() throws Exception {
        List<Model> models = List.of(
                new Model("distilbart-cnn-12-6 (1,024-token input)", () -> BartSummarizer.distilBartCnn()
                        .sessionOptions(o -> o.setIntraOpNumThreads(THREADS)).build()),
                new Model("flan-t5-base (512-token input)", () -> FlanT5TextGenerator.flanT5Base()
                        .sessionOptions(o -> o.setIntraOpNumThreads(THREADS)).build()));

        StringBuilder report = new StringBuilder()
                .append("# Long-document summarization\n\n")
                .append(String.format(Locale.ROOT, "Threads: %d. maxNewTokens: the default (256).%n%n", THREADS));

        for (Corpus corpus : List.of(BILLSUM, GOV_REPORT)) {
            List<Document> documents = documents(corpus);
            report.append("# ").append(corpus.name()).append("\n\n")
                    .append(String.format(Locale.ROOT, "%d documents, %d words on average (reference summaries: "
                                    + "%d).%n%n", documents.size(),
                            documents.stream().mapToInt(d -> words(d.text())).sum() / documents.size(),
                            documents.stream().mapToInt(d -> words(d.reference())).sum() / documents.size()));
            evaluate(models, documents, report);
        }
        report.append("""
                ROUGE-1/2/L: F1 of unigram, bigram and longest-common-subsequence overlap with the reference \
                summary, 0–100, averaged over documents; lower-cased, no stemming (rouge_score's tokenizer). \
                R-1 P / R: ROUGE-1 precision and recall; longer output raises recall and lowers precision. \
                Summary words: average length of the output.

                Data: BillSum test split (CC0-1.0) and GovReport test split (CC-BY-4.0, Huang et al., 2021, \
                "Efficient Attentions for Long Document Summarization"), downloaded at run time.
                """);

        Files.createDirectories(REPORT.getParent());
        Files.writeString(REPORT, report);
        System.out.println(report);
        System.out.println("Report written to " + REPORT.toAbsolutePath());
    }

    private static void evaluate(List<Model> models, List<Document> documents, StringBuilder report)
            throws Exception {
        for (Model model : models) {
            int limit;
            try (Summarizer summarizer = model.build().get()) {
                limit = summarizer.summarizationLimits().maxInputTokens();
            }
            report.append("## ").append(model.name()).append("\n\n")
                    .append("| Configuration | ROUGE-1 | ROUGE-2 | ROUGE-L | R-1 P / R | Summary words | Calls "
                            + "| Time |\n")
                    .append("|---|---|---|---|---|---|---|---|\n");
            for (Config config : List.of(
                    new Config("plain (truncates at the input limit)", Mode.PLAIN, null),
                    new Config("map-reduce, default chunks (" + limit + " tokens)", Mode.MAP_REDUCE, null),
                    new Config("map-reduce, " + limit / 2 + "-token chunks", Mode.MAP_REDUCE, limit / 2),
                    new Config("ChunkedSummarizer: chunk summaries joined, no reduce", Mode.CHUNKED, null))) {
                String row = run(config, model, documents);
                report.append(row);
                System.out.print(model.name() + " " + row);
            }
            report.append('\n');
        }
    }

    private static String run(Config config, Model model, List<Document> documents) throws Exception {
        List<String> references = new ArrayList<>();
        List<String> summaries = new ArrayList<>();
        long start = System.nanoTime();
        CountingSummarizer counting = new CountingSummarizer(model.build().get());
        Summarizer summarizer = switch (config.mode()) {
            case PLAIN -> counting;
            case CHUNKED -> new ChunkedSummarizer(counting);
            case MAP_REDUCE -> config.maxChunkTokens() == null ? new MapReduceSummarizer(counting)
                    : new MapReduceSummarizer(counting, config.maxChunkTokens());
        };
        try (summarizer) {
            for (Document document : documents) {
                references.add(document.reference());
                summaries.add(summarizer.summarize(document.text()).strip());
            }
        }
        long seconds = (System.nanoTime() - start) / 1_000_000_000L;
        Rouge.Scores scores = Rouge.mean(references, summaries);
        return String.format(Locale.ROOT, "| %s | **%.1f** | %.1f | %.1f | %.0f / %.0f | %d | %d | %ds |%n",
                config.name(), scores.rouge1(), scores.rouge2(), scores.rougeL(), scores.rouge1Precision(),
                scores.rouge1Recall(),
                summaries.stream().mapToInt(LongDocumentSummarizationEvaluation::words).sum() / summaries.size(),
                counting.calls, seconds);
    }

    private static int words(String text) {
        return text.isBlank() ? 0 : text.strip().split("\\s+").length;
    }

    /** Joins the line wraps inside each paragraph; paragraphs stay apart. */
    private static String clean(String text) {
        List<String> paragraphs = new ArrayList<>();
        for (String paragraph : text.strip().split("\\n\\s*\\n")) {
            paragraphs.add(paragraph.strip().replaceAll("\\s+", " "));
        }
        return String.join("\n\n", paragraphs);
    }

    private static List<Document> documents(Corpus corpus) throws IOException, InterruptedException {
        List<Document> documents = new ArrayList<>();
        ObjectMapper mapper = new ObjectMapper();
        for (int offset = 0; documents.size() < corpus.documents(); offset += 100) {
            JsonNode rows = mapper.readTree(page(corpus, offset).toFile()).get("rows");
            if (rows == null || rows.isEmpty()) {
                break;
            }
            for (JsonNode row : rows) {
                String text = row.get("row").get(corpus.textField()).asText();
                int words = words(text);
                if (words >= corpus.minWords() && words <= corpus.maxWords()
                        && documents.size() < corpus.documents()) {
                    documents.add(new Document(clean(text), row.get("row").get("summary").asText().strip()));
                }
            }
        }
        return documents;
    }

    private static Path page(Corpus corpus, int offset) throws IOException, InterruptedException {
        Path dir = DATA_ROOT.resolve(corpus.cacheDir());
        Path page = dir.resolve("test-" + offset + ".json");
        if (Files.exists(page)) {
            return page;
        }
        Files.createDirectories(dir);
        URI uri = URI.create(corpus.rowsUrl() + offset);
        System.out.println("Downloading " + corpus.name() + " rows from " + uri);
        HttpResponse<Path> response = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build()
                .send(HttpRequest.newBuilder(uri).build(), HttpResponse.BodyHandlers.ofFile(page));
        if (response.statusCode() != 200) {
            Files.deleteIfExists(page);
            throw new IOException("Failed to download " + uri + ": HTTP " + response.statusCode());
        }
        return page;
    }
}
