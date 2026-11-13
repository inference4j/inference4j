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

package io.github.inference4j.langchain4j;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentTransformer;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.segment.TextSegmentTransformer;
import io.github.inference4j.nlp.NamedEntity;
import io.github.inference4j.nlp.NamedEntityRecognizer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Replaces personal data in LangChain4j documents and text segments with placeholders such as
 * {@code [EMAIL]}, in-process, before the text is embedded, stored or sent to a hosted model.
 *
 * <pre>{@code
 * try (var redactor = Inference4jPiiRedactor.builder()
 *         .recognizer(GlinerEntityRecognizer.pii().threshold(0.3f).stride(64).build())
 *         .build()) {
 *
 *     EmbeddingStoreIngestor.builder()
 *             .documentTransformer(redactor)    // or .textSegmentTransformer(redactor.asTextSegmentTransformer())
 *             .embeddingModel(embeddingModel)
 *             .embeddingStore(store)
 *             .build()
 *             .ingest(documents);
 * }
 * }</pre>
 *
 * <p><b>Redaction reduces exposure; it does not guarantee that no personal data remains.</b> Detection is
 * probabilistic and no model finds every entity. For {@code GlinerEntityRecognizer.pii()}:
 * <ul>
 *   <li>a lower {@code threshold} (the model card suggests 0.3) finds more entities, at the cost of some
 *       false positives;</li>
 *   <li>text longer than the model reads (2048 words) is truncated by default, and the rest is <b>not</b>
 *       redacted. Build the recognizer with {@code stride(...)} so long text is read in full.</li>
 * </ul>
 *
 * <p>Spans that overlap or touch, and spans with the same label separated only by whitespace (such as
 * "John" and "Smith"), become one placeholder. A merged span takes the label of its highest-scoring part.
 *
 * <p>Redacted documents and segments keep their metadata and gain {@value #REDACTED_COUNT} (the number of
 * placeholders, {@code 0} when nothing was found) plus one {@code pii_redacted_<label>} count per label
 * found, for example {@code pii_redacted_email} or {@code pii_redacted_phone_number}.
 *
 * <p>The redactor owns the recognizer: {@link #close()} closes it. Thread-safe: one redactor can serve a whole
 * application.
 */
public class Inference4jPiiRedactor implements DocumentTransformer, AutoCloseable {

    /** Number of placeholders written into the text ({@code int}). */
    public static final String REDACTED_COUNT = "pii_redacted_count";

    /** Prefix of the per-label counts, followed by the label in snake case ({@code int}). */
    public static final String REDACTED_PREFIX = "pii_redacted_";

    private final NamedEntityRecognizer recognizer;
    private final Set<String> labels;          // lower case; null means every label
    private final Float minScore;
    private final Function<NamedEntity, String> placeholder;
    private final TextSegmentTransformer segmentTransformer = this::transform;

    private Inference4jPiiRedactor(Builder builder) {
        this.recognizer = Objects.requireNonNull(builder.recognizer, "recognizer is required");
        this.labels = builder.labels == null ? null : builder.labels.stream()
                .map(label -> label.toLowerCase(Locale.ROOT))
                .collect(Collectors.toUnmodifiableSet());
        this.minScore = builder.minScore;
        this.placeholder = builder.placeholder;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Returns the document with personal data replaced by placeholders, its metadata kept and the redaction
     * counts added.
     */
    @Override
    public Document transform(Document document) {
        Redaction redaction = redactWithCounts(document.text());
        return Document.from(redaction.text(), redaction.addCounts(document.metadata().copy()));
    }

    /**
     * Returns the segment with personal data replaced by placeholders, its metadata kept and the redaction
     * counts added.
     */
    public TextSegment transform(TextSegment segment) {
        Redaction redaction = redactWithCounts(segment.text());
        return TextSegment.from(redaction.text(), redaction.addCounts(segment.metadata().copy()));
    }

    /**
     * This redactor as a {@link TextSegmentTransformer}, for {@code EmbeddingStoreIngestor.textSegmentTransformer}.
     * It shares this redactor's recognizer; closing the redactor makes it unusable.
     *
     * <p>A separate view because one class cannot implement both {@link DocumentTransformer} and
     * {@link TextSegmentTransformer}: their {@code transformAll} methods clash.
     */
    public TextSegmentTransformer asTextSegmentTransformer() {
        return segmentTransformer;
    }

    /**
     * Returns the text with personal data replaced by placeholders.
     */
    public String redact(String text) {
        return redactWithCounts(text).text();
    }

    /**
     * Closes the recognizer.
     */
    @Override
    public void close() {
        recognizer.close();
    }

    private Redaction redactWithCounts(String text) {
        if (text == null || text.isEmpty()) {
            return new Redaction(text, List.of());
        }
        List<NamedEntity> spans = merge(text, selected(text));
        StringBuilder redacted = new StringBuilder(text);
        for (int i = spans.size() - 1; i >= 0; i--) {                 // back to front keeps earlier offsets valid
            NamedEntity span = spans.get(i);
            redacted.replace(span.start(), span.end(), placeholder.apply(span));
        }
        return new Redaction(redacted.toString(), spans);
    }

    /** Entities that pass the label and score filters and lie inside the text, ordered by position. */
    private List<NamedEntity> selected(String text) {
        List<NamedEntity> kept = new ArrayList<>();
        for (NamedEntity entity : recognizer.recognize(text)) {
            if (labels != null && !labels.contains(entity.label().toLowerCase(Locale.ROOT))) {
                continue;
            }
            if (minScore != null && entity.score() < minScore) {
                continue;
            }
            if (entity.start() < 0 || entity.end() > text.length() || entity.start() >= entity.end()) {
                continue;
            }
            kept.add(entity);
        }
        kept.sort(Comparator.comparingInt(NamedEntity::start).thenComparing(NamedEntity::end, Comparator.reverseOrder()));
        return kept;
    }

    /**
     * Joins spans that overlap or touch, and same-label spans separated only by whitespace. The joined span
     * takes the label and score of its highest-scoring part.
     */
    static List<NamedEntity> merge(String text, List<NamedEntity> sorted) {
        List<NamedEntity> merged = new ArrayList<>();
        for (NamedEntity next : sorted) {
            if (!merged.isEmpty()) {
                NamedEntity last = merged.get(merged.size() - 1);
                boolean joins = next.start() <= last.end()
                        || (next.label().equalsIgnoreCase(last.label())
                            && text.substring(last.end(), next.start()).isBlank());
                if (joins) {
                    int start = last.start();
                    int end = Math.max(last.end(), next.end());
                    NamedEntity best = next.score() > last.score() ? next : last;
                    merged.set(merged.size() - 1,
                            new NamedEntity(text.substring(start, end), best.label(), start, end, best.score()));
                    continue;
                }
            }
            merged.add(next);
        }
        return merged;
    }

    static String snakeCase(String label) {
        String snake = label.strip().toLowerCase(Locale.ROOT).replaceAll("[^\\p{Alnum}]+", "_");
        return snake.replaceAll("^_+|_+$", "");
    }

    private record Redaction(String text, List<NamedEntity> spans) {

        Metadata addCounts(Metadata metadata) {
            Map<String, Integer> perLabel = new TreeMap<>();
            for (NamedEntity span : spans) {
                perLabel.merge(snakeCase(span.label()), 1, Integer::sum);
            }
            metadata.put(REDACTED_COUNT, spans.size());
            perLabel.forEach((label, count) -> metadata.put(REDACTED_PREFIX + label, count));
            return metadata;
        }
    }

    /**
     * Builder for {@link Inference4jPiiRedactor}.
     */
    public static class Builder {

        private NamedEntityRecognizer recognizer;
        private Collection<String> labels;
        private Float minScore;
        private Function<NamedEntity, String> placeholder =
                entity -> "[" + snakeCase(entity.label()).toUpperCase(Locale.ROOT) + "]";

        private Builder() {
        }

        /**
         * The recognizer that finds personal data. Required. Typically
         * {@code GlinerEntityRecognizer.pii().threshold(0.3f).stride(64).build()}; any
         * {@link NamedEntityRecognizer}, such as {@code BertNerRecognizer}, works.
         */
        public Builder recognizer(NamedEntityRecognizer recognizer) {
            this.recognizer = recognizer;
            return this;
        }

        /**
         * Redacts only entities with these labels (case-insensitive), such as {@code "email"} and
         * {@code "phone number"}. By default every entity the recognizer finds is redacted.
         */
        public Builder labels(Collection<String> labels) {
            this.labels = List.copyOf(labels);
            return this;
        }

        /**
         * Redacts only entities scoring at least {@code minScore}, on top of the recognizer's own threshold.
         * Unset by default. Lowering the recognizer's threshold is the way to find more entities.
         */
        public Builder minScore(float minScore) {
            this.minScore = minScore;
            return this;
        }

        /**
         * What replaces each span. Defaults to the label in upper snake case in brackets:
         * {@code [EMAIL]}, {@code [PHONE_NUMBER]}.
         */
        public Builder placeholder(Function<NamedEntity, String> placeholder) {
            this.placeholder = Objects.requireNonNull(placeholder, "placeholder");
            return this;
        }

        /**
         * @throws NullPointerException if no recognizer is set
         */
        public Inference4jPiiRedactor build() {
            return new Inference4jPiiRedactor(this);
        }
    }
}
