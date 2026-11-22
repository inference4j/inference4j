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
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Evaluates {@link ChunkedTranslator} on long English documents translated into German, against plain
 * translation (which truncates at the model's limit) and sentence-by-sentence translation.
 *
 * <p>Opt-in, never run in CI: set {@code INFERENCE4J_EVAL=true}.
 * <pre>
 * INFERENCE4J_EVAL=true ./gradlew :inference4j-core:modelTest --tests '*LongDocumentTranslationEvaluation'
 * </pre>
 * The report is written to {@code inference4j-core/build/reports/evaluation/long-translation.md}.
 *
 * <p><b>Data</b>, read through the Hugging Face datasets-server, sentences in document order:
 * <ul>
 *   <li>MarianMT (opus-mt-en-es): News Commentary v16 en-es (Project Syndicate articles). It is part of
 *       the OPUS data opus-mt was trained on, so absolute scores are inflated; compare configurations only.</li>
 *   <li>Flan-T5 (en→de): WMT19 de-en validation (newstest2018).</li>
 * </ul>
 * Consecutive sentences are grouped into documents of {@value #SENTENCES_PER_DOCUMENT} sentences, with a
 * paragraph break every {@value #SENTENCES_PER_PARAGRAPH}. The text is for research use only. It is
 * downloaded at run time into {@code ~/.cache/inference4j/datasets} and must never be committed or
 * redistributed.
 */
@EnabledIfEnvironmentVariable(named = "INFERENCE4J_EVAL", matches = "true")
class LongDocumentTranslationEvaluation {

    private static final Path DATA_ROOT = Path.of(System.getProperty("user.home"),
            ".cache", "inference4j", "datasets");

    /** A parallel corpus on the datasets-server: rows hold {@code translation.<source>} and {@code .<target>}. */
    record Corpus(String name, String rowsUrl, String cacheDir, String source, String target) {
    }

    private static final Corpus NEWSTEST_2018 = new Corpus("newstest2018 en→de",
            "https://datasets-server.huggingface.co/rows?dataset=wmt%2Fwmt19&config=de-en&split=validation"
                    + "&length=100&offset=", "wmt19-de-en", "en", "de");
    private static final Corpus NEWS_COMMENTARY = new Corpus("News Commentary en→es",
            "https://datasets-server.huggingface.co/rows?dataset=Helsinki-NLP%2Fnews_commentary&config=en-es"
                    + "&split=train&length=100&offset=", "news-commentary-en-es", "en", "es");
    private static final Path REPORT = Path.of("build", "reports", "evaluation", "long-translation.md");

    private static final int DOCUMENTS = 16;
    private static final int SENTENCES_PER_DOCUMENT = 24;
    private static final int SENTENCES_PER_PARAGRAPH = 4;
    private static final int THREADS = 4;
    /** Long output limit; MarianMT's decoder has 512 positions, so 512 new tokens would overrun it. */
    private static final int LONG_OUTPUT = 500;
    /** Prefix length for the boundary check: short enough for one plain call. */
    private static final int PREFIX_TOKENS = 480;

    /** A document: its source sentences and their reference translations. */
    record Document(List<String> sourceSentences, List<String> referenceSentences) {

        String source() {
            return join(sourceSentences);
        }

        String reference() {
            return join(referenceSentences);
        }

        Document prefix(int sentences) {
            return new Document(sourceSentences.subList(0, sentences), referenceSentences.subList(0, sentences));
        }
    }

    enum Mode { PLAIN, CHUNKED, SENTENCES }

    /** One way of translating; {@code maxChunkTokens} is {@code null} for the decorator's default. */
    record Config(String name, Mode mode, Integer maxChunkTokens, Supplier<Translator> model) {
    }

    /** Counts calls, and calls whose output reached the output limit. */
    static final class CountingTranslator implements Translator {
        private final Translator delegate;
        int calls;
        int cutOff;

        CountingTranslator(Translator delegate) {
            this.delegate = delegate;
        }

        private GenerationResult count(GenerationResult result, int maxOutputTokens) {
            calls++;
            if (result.generatedTokens() >= maxOutputTokens) {
                cutOff++;
            }
            return result;
        }

        @Override
        public GenerationResult translate(String text, Consumer<String> tokenListener) {
            return count(delegate.translate(text, tokenListener), delegate.translationLimits().maxOutputTokens());
        }

        @Override
        public GenerationResult translate(String text, Language source, Language target,
                                          Consumer<String> tokenListener) {
            return count(delegate.translate(text, source, target, tokenListener),
                    delegate.translationLimits(source, target).maxOutputTokens());
        }

        @Override
        public TokenLimits translationLimits() {
            return delegate.translationLimits();
        }

        @Override
        public TokenLimits translationLimits(Language source, Language target) {
            return delegate.translationLimits(source, target);
        }

        @Override
        public void close() throws Exception {
            delegate.close();
        }
    }

    /**
     * A model and the corpus it is scored on. {@code pair} is set for models that take the language pair
     * per call (Flan-T5, English to German); others have it built in.
     */
    record Model(String name, Corpus corpus, Supplier<Translator> plain, Supplier<Translator> longOutput,
                 boolean pair) {
    }

    @Test
    void evaluateOnLongDocuments() throws Exception {
        Model marian = new Model("opus-mt-en-es", NEWS_COMMENTARY,
                () -> MarianTranslator.builder().modelId("inference4j/opus-mt-en-es")
                        .sessionOptions(o -> o.setIntraOpNumThreads(THREADS)).build(),
                () -> MarianTranslator.builder().modelId("inference4j/opus-mt-en-es").maxNewTokens(LONG_OUTPUT)
                        .sessionOptions(o -> o.setIntraOpNumThreads(THREADS)).build(),
                false);
        Model flan = new Model("flan-t5-base", NEWSTEST_2018,
                () -> FlanT5TextGenerator.flanT5Base()
                        .sessionOptions(o -> o.setIntraOpNumThreads(THREADS)).build(),
                () -> FlanT5TextGenerator.flanT5Base().maxNewTokens(LONG_OUTPUT)
                        .sessionOptions(o -> o.setIntraOpNumThreads(THREADS)).build(),
                true);

        StringBuilder report = new StringBuilder()
                .append("# Long-document translation\n\n")
                .append(String.format(Locale.ROOT, "Documents of %d consecutive sentences, a paragraph break "
                        + "every %d sentences. Threads: %d.%n%n", SENTENCES_PER_DOCUMENT, SENTENCES_PER_PARAGRAPH,
                        THREADS));

        for (Model model : List.of(marian, flan)) {
            List<Document> documents = documents(model.corpus());
            List<Config> configs = List.of(
                    new Config("plain (truncates at the input limit), maxNewTokens 500", Mode.PLAIN, null,
                            model.longOutput()),
                    new Config("chunked, defaults (maxNewTokens 256, 64-token chunks)", Mode.CHUNKED, null,
                            model.plain()),
                    new Config("chunked, 64-token chunks, maxNewTokens 500", Mode.CHUNKED, 64, model.longOutput()),
                    new Config("chunked, 128-token chunks, maxNewTokens 500", Mode.CHUNKED, 128, model.longOutput()),
                    new Config("chunked, 256-token chunks, maxNewTokens 500", Mode.CHUNKED, 256, model.longOutput()),
                    new Config("chunked, 512-token chunks, maxNewTokens 500", Mode.CHUNKED, 512, model.longOutput()),
                    new Config("reference: one call per sentence", Mode.SENTENCES, null, model.plain()));

            report.append("## ").append(model.name()).append(", ").append(model.corpus().name()).append("\n\n")
                    .append(String.format(Locale.ROOT, "%d documents, %d words on average.%n%n", documents.size(),
                            documents.stream().mapToInt(d -> words(d.source())).sum() / documents.size()))
                    .append("| Configuration | chrF | Source covered | Length ratio | Calls | Calls cut off | Time |\n")
                    .append("|---|---|---|---|---|---|---|\n");
            for (Config config : configs) {
                String row = run(config, model.pair(), documents);
                report.append(row);
                System.out.print(model.name() + " " + row);
            }

            // Boundary check: on prefixes short enough for one call, chunking only adds boundaries
            List<Document> prefixes = prefixes(documents, model);
            report.append(String.format(Locale.ROOT, "%nBoundary check, on each document's first sentences "
                            + "up to %d tokens (one plain call, no truncation):%n%n", PREFIX_TOKENS))
                    .append("| Configuration | chrF | Source covered | Length ratio | Calls | Calls cut off | Time |\n")
                    .append("|---|---|---|---|---|---|---|\n");
            for (Config config : List.of(
                    new Config("plain, maxNewTokens 500", Mode.PLAIN, null, model.longOutput()),
                    new Config("chunked, 128-token chunks, maxNewTokens 500", Mode.CHUNKED, 128, model.longOutput()),
                    new Config("chunked, 64-token chunks, maxNewTokens 500", Mode.CHUNKED, 64, model.longOutput()))) {
                report.append(run(config, model.pair(), prefixes));
            }
            report.append('\n');
        }
        report.append("""
                chrF: corpus-level character n-gram F-score (orders 1–6, recall weighted ×2, sacreBLEU chrF2 style),                 0–100, against the full reference. Source covered: share of source tokens the model saw.                 Length ratio: output characters / reference characters. Calls cut off: calls whose output reached                 maxNewTokens.

                Data: News Commentary v16 en-es (in opus-mt's training data, so MarianMT scores are inflated; \
                compare configurations only) and WMT19 de-en validation (newstest2018). Research use only, \
                downloaded at run time.
                """);

        Files.createDirectories(REPORT.getParent());
        Files.writeString(REPORT, report);
        System.out.println(report);
        System.out.println("Report written to " + REPORT.toAbsolutePath());
    }

    private static String run(Config config, boolean pair, List<Document> documents) throws Exception {
        List<String> references = new ArrayList<>();
        List<String> hypotheses = new ArrayList<>();
        double covered = 0;
        long start = System.nanoTime();
        CountingTranslator counting = new CountingTranslator(config.model().get());
        Translator translator = config.mode() != Mode.CHUNKED ? counting
                : config.maxChunkTokens() == null ? new ChunkedTranslator(counting)
                : new ChunkedTranslator(counting, config.maxChunkTokens());
        try (translator) {
            TokenLimits limits = pair ? counting.translationLimits(Language.EN, Language.DE)
                    : counting.translationLimits();
            Function<String, String> translate = text -> pair
                    ? translator.translate(text, Language.EN, Language.DE)
                    : translator.translate(text);
            for (Document document : documents) {
                references.add(document.reference());
                if (config.mode() == Mode.SENTENCES) {
                    hypotheses.add(join(document.sourceSentences().stream().map(translate).toList()));
                    covered += 1;
                } else {
                    hypotheses.add(translate.apply(document.source()));
                    covered += config.mode() == Mode.PLAIN
                            ? Math.min(1.0, (double) limits.maxInputTokens()
                                / limits.countInputTokens(document.source()))
                            : 1;
                }
            }
        }
        long seconds = (System.nanoTime() - start) / 1_000_000_000L;
        double outputChars = hypotheses.stream().mapToInt(String::length).sum();
        double referenceChars = references.stream().mapToInt(String::length).sum();
        return String.format(Locale.ROOT, "| %s | **%.1f** | %.0f%% | %.2f | %d | %d | %ds |%n",
                config.name(), ChrF.corpus(references, hypotheses), 100 * covered / documents.size(),
                outputChars / referenceChars, counting.calls, counting.cutOff, seconds);
    }

    private static List<Document> prefixes(List<Document> documents, Model model) throws Exception {
        List<Document> prefixes = new ArrayList<>();
        try (Translator translator = model.plain().get()) {
            TokenLimits limits = model.pair() ? translator.translationLimits(Language.EN, Language.DE)
                    : translator.translationLimits();
            for (Document document : documents) {
                int sentences = 1;
                while (sentences < document.sourceSentences().size()
                        && limits.countInputTokens(document.prefix(sentences + 1).source()) <= PREFIX_TOKENS) {
                    sentences++;
                }
                prefixes.add(document.prefix(sentences));
            }
        }
        return prefixes;
    }

    private static String join(List<String> sentences) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < sentences.size(); i++) {
            if (i > 0) {
                text.append(i % SENTENCES_PER_PARAGRAPH == 0 ? "\n\n" : " ");
            }
            text.append(sentences.get(i).strip());
        }
        return text.toString();
    }

    private static int words(String text) {
        return text.strip().split("\\s+").length;
    }

    private static List<Document> documents(Corpus corpus) throws IOException, InterruptedException {
        List<String> english = new ArrayList<>();
        List<String> german = new ArrayList<>();
        ObjectMapper mapper = new ObjectMapper();
        for (int offset = 0; offset < DOCUMENTS * SENTENCES_PER_DOCUMENT; offset += 100) {
            for (JsonNode row : mapper.readTree(page(corpus, offset).toFile()).get("rows")) {
                JsonNode translation = row.get("row").get("translation");
                english.add(translation.get(corpus.source()).asText());
                german.add(translation.get(corpus.target()).asText());
            }
        }
        List<Document> documents = new ArrayList<>();
        for (int d = 0; d < DOCUMENTS; d++) {
            int from = d * SENTENCES_PER_DOCUMENT;
            documents.add(new Document(english.subList(from, from + SENTENCES_PER_DOCUMENT),
                    german.subList(from, from + SENTENCES_PER_DOCUMENT)));
        }
        return documents;
    }

    private static Path page(Corpus corpus, int offset) throws IOException, InterruptedException {
        Path dir = DATA_ROOT.resolve(corpus.cacheDir());
        Path page = dir.resolve("rows-" + offset + ".json");
        if (Files.exists(page)) {
            return page;
        }
        Files.createDirectories(dir);
        URI uri = URI.create(corpus.rowsUrl() + offset);
        System.out.println("Downloading " + corpus.name() + " rows (research use only) from " + uri);
        HttpResponse<Path> response = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build()
                .send(HttpRequest.newBuilder(uri).build(), HttpResponse.BodyHandlers.ofFile(page));
        if (response.statusCode() != 200) {
            Files.deleteIfExists(page);
            throw new IOException("Failed to download " + uri + ": HTTP " + response.statusCode());
        }
        return page;
    }
}
