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

import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.tokenizer.WordPieceTokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.function.Supplier;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Evaluates {@link BertNerRecognizer} on whole news articles from the CoNLL-2003 test set, the
 * benchmark {@code distilbert-NER} was trained and evaluated on, comparing plain recognition with
 * {@code stride} windowing at several window sizes.
 *
 * <p>Opt-in, never run in CI: set {@code INFERENCE4J_EVAL=true}.
 * <pre>
 * INFERENCE4J_EVAL=true ./gradlew :inference4j-core:modelTest --tests '*BertNerLongDocumentEvaluation'
 * </pre>
 * The report is written to {@code inference4j-core/build/reports/evaluation/ner-long-documents.md}.
 *
 * <p><b>Data license:</b> CoNLL-2003 annotations come from the University of Antwerp; the article
 * text is from the Reuters Corpus and is available for research use only. The data is downloaded
 * at run time into {@code ~/.cache/inference4j/datasets/conll2003} and must never be committed or
 * redistributed.
 */
@EnabledIfEnvironmentVariable(named = "INFERENCE4J_EVAL", matches = "true")
class BertNerLongDocumentEvaluation {

    // Same archive the Hugging Face "conll2003" dataset loader downloads
    private static final URI DATA_URL = URI.create("https://data.deepai.org/conll2003.zip");
    private static final Path DATA_DIR = Path.of(System.getProperty("user.home"),
            ".cache", "inference4j", "datasets", "conll2003");
    private static final Path REPORT = Path.of("build", "reports", "evaluation", "ner-long-documents.md");

    /** One article: its text, its sentences' offsets, and the gold entities with character offsets. */
    record Document(String text, List<int[]> sentences, Set<Span> entities) {
    }

    /** An entity as a scoring unit: exact span plus type. */
    record Span(int start, int end, String label) {
    }

    /** How the text is fed to the recognizer. */
    enum Mode {
        /** The whole article in one call (truncated or windowed by the recognizer). */
        WHOLE_TEXT,
        /** One call per sentence, using the dataset's human-marked sentence boundaries. */
        GOLD_SENTENCES,
        /** One call per sentence, as found by the JDK's rule-based {@link BreakIterator}. */
        BREAK_ITERATOR_SENTENCES
    }

    /** One way of running the recognizer under evaluation. */
    record Config(String name, Supplier<BertNerRecognizer> factory, Mode mode) {
    }

    @Test
    void evaluateOnCoNLL2003TestArticles() throws Exception {
        List<Document> documents = parse(testFile());
        WordPieceTokenizer tokenizer = WordPieceTokenizer.fromVocabFile(
                HuggingFaceModelSource.defaultInstance()
                        .resolve("inference4j/distilbert-NER", List.of("model.onnx", "vocab.txt", "config.json"))
                        .resolve("vocab.txt"), false);
        List<Document> longDocuments = documents.stream()
                .filter(d -> tokenizer.encode(d.text(), Integer.MAX_VALUE).inputIds().length > 512)
                .toList();

        List<Config> configs = List.of(
                new Config("no windowing (truncates at 512)",
                        () -> BertNerRecognizer.builder().build(), Mode.WHOLE_TEXT),
                new Config("maxLength 512, stride 128",
                        () -> BertNerRecognizer.builder().stride(128).build(), Mode.WHOLE_TEXT),
                new Config("maxLength 256, stride 64",
                        () -> BertNerRecognizer.builder().maxLength(256).stride(64).build(), Mode.WHOLE_TEXT),
                new Config("maxLength 128, stride 32",
                        () -> BertNerRecognizer.builder().maxLength(128).stride(32).build(), Mode.WHOLE_TEXT),
                new Config("sentence by sentence, BreakIterator splitting",
                        () -> BertNerRecognizer.builder().build(), Mode.BREAK_ITERATOR_SENTENCES),
                new Config("reference: sentence by sentence, human-marked sentences",
                        () -> BertNerRecognizer.builder().build(), Mode.GOLD_SENTENCES));

        StringBuilder report = new StringBuilder()
                .append("# NER on long documents: CoNLL-2003 test articles\n\n")
                .append("Model: `inference4j/distilbert-NER`. Entity-level scores: an entity counts only if its exact ")
                .append("span **and** type match the human annotation.\n\n")
                .append(String.format(Locale.ROOT, "- All articles: %d (%d gold entities)%n", documents.size(), goldCount(documents)))
                .append(String.format(Locale.ROOT, "- Long articles (over 512 tokens): %d (%d gold entities)%n%n",
                        longDocuments.size(), goldCount(longDocuments)))
                .append("| Configuration | All: P | All: R | All: F1 | Long: P | Long: R | Long: F1 | Time |\n")
                .append("|---|---|---|---|---|---|---|---|\n");

        for (Config config : configs) {
            long start = System.nanoTime();
            Score all;
            Score longOnly;
            try (BertNerRecognizer ner = config.factory().get()) {
                all = score(ner, documents, config.mode());
                longOnly = score(ner, longDocuments, config.mode());
            }
            long seconds = (System.nanoTime() - start) / 1_000_000_000L;
            report.append(String.format(Locale.ROOT, "| %s | %.3f | %.3f | **%.3f** | %.3f | %.3f | **%.3f** | %ds |%n",
                    config.name(), all.precision(), all.recall(), all.f1(),
                    longOnly.precision(), longOnly.recall(), longOnly.f1(), seconds));
        }
        report.append("\nP = precision (found entities that are correct), R = recall (correct entities that were found), ")
                .append("F1 = their harmonic mean. Time covers scoring both sets.\n\n")
                .append("Data: CoNLL-2003 (Reuters, research use only), downloaded to `")
                .append(DATA_DIR).append("`; not redistributed.\n");

        Files.createDirectories(REPORT.getParent());
        Files.writeString(REPORT, report);
        System.out.println(report);
        System.out.println("Report written to " + REPORT.toAbsolutePath());
    }

    // --- scoring ---

    record Score(int truePositives, int predicted, int gold) {
        double precision() {
            return predicted == 0 ? 0 : (double) truePositives / predicted;
        }

        double recall() {
            return gold == 0 ? 0 : (double) truePositives / gold;
        }

        double f1() {
            double p = precision();
            double r = recall();
            return p + r == 0 ? 0 : 2 * p * r / (p + r);
        }
    }

    private static Score score(BertNerRecognizer ner, List<Document> documents, Mode mode) {
        int truePositives = 0;
        int predicted = 0;
        int gold = 0;
        for (Document document : documents) {
            Set<Span> found = new HashSet<>();
            if (mode != Mode.WHOLE_TEXT) {
                // How CoNLL is normally scored: each sentence is recognized on its own
                List<int[]> sentences = mode == Mode.GOLD_SENTENCES
                        ? document.sentences()
                        : breakIteratorSentences(document.text());
                for (int[] sentence : sentences) {
                    String text = document.text().substring(sentence[0], sentence[1]);
                    for (NamedEntity e : ner.recognize(text)) {
                        found.add(new Span(sentence[0] + e.start(), sentence[0] + e.end(), e.label()));
                    }
                }
            } else {
                for (NamedEntity e : ner.recognize(document.text())) {
                    found.add(new Span(e.start(), e.end(), e.label()));
                }
            }
            predicted += found.size();
            gold += document.entities().size();
            found.retainAll(document.entities());
            truePositives += found.size();
        }
        return new Score(truePositives, predicted, gold);
    }

    /** Sentence spans found by the JDK's rule-based splitter (Unicode UAX #29 rules). */
    static List<int[]> breakIteratorSentences(String text) {
        BreakIterator iterator = BreakIterator.getSentenceInstance(Locale.ENGLISH);
        iterator.setText(text);
        List<int[]> sentences = new ArrayList<>();
        for (int start = iterator.first(), end = iterator.next();
             end != BreakIterator.DONE;
             start = end, end = iterator.next()) {
            String sentence = text.substring(start, end);
            int leading = sentence.length() - sentence.stripLeading().length();
            int trailing = sentence.length() - sentence.stripTrailing().length();
            if (start + leading < end - trailing) {
                sentences.add(new int[]{start + leading, end - trailing});
            }
        }
        return sentences;
    }

    private static int goldCount(List<Document> documents) {
        return documents.stream().mapToInt(d -> d.entities().size()).sum();
    }

    // --- data ---

    /**
     * Parses the CoNLL format: one token per line ({@code word POS chunk NER-tag}), blank lines between
     * sentences, {@code -DOCSTART-} between articles. Tokens are joined with single spaces, which
     * matches how the recognizer splits words, so gold offsets line up with predicted ones.
     */
    static List<Document> parse(Path file) throws IOException {
        List<Document> documents = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        List<int[]> sentences = new ArrayList<>();
        int sentenceStart = -1;
        Set<Span> entities = new HashSet<>();
        int entityStart = -1;
        int entityEnd = -1;
        String entityLabel = null;

        for (String line : Files.readAllLines(file)) {
            if (line.startsWith("-DOCSTART-")) {
                if (entityLabel != null) {
                    entities.add(new Span(entityStart, entityEnd, entityLabel));
                    entityLabel = null;
                }
                if (sentenceStart >= 0) {
                    sentences.add(new int[]{sentenceStart, text.length()});
                    sentenceStart = -1;
                }
                if (text.length() > 0) {
                    documents.add(new Document(text.toString(), List.copyOf(sentences), Set.copyOf(entities)));
                }
                text.setLength(0);
                sentences.clear();
                entities.clear();
                continue;
            }
            if (line.isBlank()) {
                if (sentenceStart >= 0) {   // sentence boundary: words keep flowing with a single space
                    sentences.add(new int[]{sentenceStart, text.length()});
                    sentenceStart = -1;
                }
                continue;
            }
            String[] columns = line.trim().split(" ");
            String word = columns[0];
            String tag = columns[columns.length - 1];

            if (text.length() > 0) {
                text.append(' ');
            }
            int start = text.length();
            if (sentenceStart < 0) {
                sentenceStart = start;
            }
            text.append(word);
            int end = text.length();

            String type = tag.equals("O") ? null : tag.substring(2);
            boolean continues = entityLabel != null && tag.startsWith("I-") && type.equals(entityLabel);
            if (continues) {
                entityEnd = end;
            } else {
                if (entityLabel != null) {
                    entities.add(new Span(entityStart, entityEnd, entityLabel));
                }
                entityLabel = type;
                entityStart = start;
                entityEnd = end;
            }
        }
        if (entityLabel != null) {
            entities.add(new Span(entityStart, entityEnd, entityLabel));
        }
        if (sentenceStart >= 0) {
            sentences.add(new int[]{sentenceStart, text.length()});
        }
        if (text.length() > 0) {
            documents.add(new Document(text.toString(), List.copyOf(sentences), Set.copyOf(entities)));
        }
        return documents;
    }

    private static Path testFile() throws IOException, InterruptedException {
        Path test = DATA_DIR.resolve("test.txt");
        if (Files.exists(test)) {
            return test;
        }
        Files.createDirectories(DATA_DIR);
        System.out.println("Downloading CoNLL-2003 (research use only) to " + DATA_DIR);
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build();
        HttpResponse<InputStream> response = client.send(
                HttpRequest.newBuilder(DATA_URL).build(), HttpResponse.BodyHandlers.ofInputStream());
        if (response.statusCode() != 200) {
            throw new IOException("Failed to download " + DATA_URL + ": HTTP " + response.statusCode());
        }
        try (ZipInputStream zip = new ZipInputStream(response.body())) {
            for (ZipEntry entry; (entry = zip.getNextEntry()) != null; ) {
                if (entry.getName().endsWith("test.txt")) {
                    Files.copy(zip, test, StandardCopyOption.REPLACE_EXISTING);
                    return test;
                }
            }
        }
        throw new IOException("test.txt not found in " + DATA_URL);
    }
}
