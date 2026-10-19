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
import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.TokenWindows;
import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.tokenizer.SpecialTokens;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.TokenizerJsonParser;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Zero-shot named entity recognizer backed by <a href="https://github.com/urchade/GLiNER">GLiNER</a>.
 *
 * <p>GLiNER reads the entity types and the text together, as {@code <<ENT>> person <<ENT>> invoice id <<SEP>> text},
 * and scores every span of up to 12 words against every type. The types are plain words chosen at call time.
 * Input building and decoding follow the {@code gliner} Python package (0.2.29), whose output this class matches.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * try (var ner = GlinerEntityRecognizer.small().build()) {       // general entities, ~610 MB (default)
 *     ner.recognize(text, List.of("person", "organization", "invoice id"));
 * }
 * try (var pii = GlinerEntityRecognizer.pii().build()) {         // PII, PHI and PCI, ~665 MB
 *     pii.recognize(text);                                       // uses the preset's PII labels
 * }
 * }</pre>
 *
 * <p>{@link #recognize(String)} uses the {@linkplain Builder#labels(List) default labels}, so a GLiNER recognizer
 * can stand in for a {@link NamedEntityRecognizer} such as {@link BertNerRecognizer}.
 *
 * <h2>Decoding</h2>
 * <p>A span is a candidate when its probability is above the {@linkplain Builder#threshold(float) threshold}. The
 * candidates are taken best first, and each is kept unless it overlaps one already kept, so entities never overlap.
 * With {@link Builder#multiLabel(boolean) multiLabel(true)}, the exact same span may carry several labels.
 *
 * <h2>Long text</h2>
 * <p>Each model reads at most {@code max_len} words (384 for small, 2048 for the PII model). Longer text is handled
 * by the {@linkplain Builder#truncation(TruncationPolicy) truncation policy}, or, with {@link Builder#stride(int)},
 * read in overlapping windows of words; each entity is taken from the window where it starts in the owned part.
 *
 * <p>Detection is probabilistic: no model finds every entity, which matters for PII redaction in particular.
 *
 * @see ZeroShotEntityRecognizer
 * @see NamedEntity
 */
public class GlinerEntityRecognizer implements ZeroShotEntityRecognizer, NamedEntityRecognizer {

    private static final String SMALL = "inference4j/gliner-small-v2.1";
    private static final String PII = "inference4j/gliner-pii-base-v1.0";
    private static final List<String> GENERAL_LABELS = List.of("person", "organization", "location");
    private static final List<String> PII_LABELS = List.of(
            "person", "email", "phone number", "address", "date of birth", "credit card number",
            "bank account number", "social security number", "passport number", "ip address");
    // gliner's WhitespaceTokenSplitter: words, hyphenated or underscored words, and single symbols
    private static final Pattern WORDS = Pattern.compile("\\w+(?:[-_]\\w+)*|\\S", Pattern.UNICODE_CHARACTER_CLASS);

    private final InferenceSession session;
    private final Tokenizer tokenizer;
    private final SpecialTokens special;
    private final long entToken;
    private final long sepToken;
    private final int maxWidth;
    private final int maxWords;
    private final List<String> defaultLabels;
    private final float threshold;
    private final boolean multiLabel;
    private final TruncationGuard truncationGuard;
    private final Integer stride;

    private GlinerEntityRecognizer(Builder builder, InferenceSession session, Tokenizer tokenizer, SpecialTokens special,
                                   long entToken, long sepToken, int maxWidth, int maxWords) {
        this.session = session;
        this.tokenizer = tokenizer;
        this.special = special;
        this.entToken = entToken;
        this.sepToken = sepToken;
        this.maxWidth = maxWidth;
        this.maxWords = maxWords;
        this.defaultLabels = List.copyOf(builder.labels);
        this.threshold = builder.threshold;
        this.multiLabel = builder.multiLabel;
        this.truncationGuard = new TruncationGuard("GlinerEntityRecognizer", builder.truncation);
        this.stride = builder.stride;
    }

    /** GLiNER small v2.1 (~610 MB), the default: general entities; default labels person, organization, location. */
    public static Builder small() {
        return new Builder().modelId(SMALL).labels(GENERAL_LABELS);
    }

    /**
     * Knowledgator's GLiNER PII base (~665 MB), fine-tuned for personal, health and payment data; default labels are
     * common PII types (person, email, phone number, address, credit card number and others).
     */
    public static Builder pii() {
        return new Builder().modelId(PII).labels(PII_LABELS);
    }

    /** A builder for the default model, {@link #small()}. */
    public static Builder builder() {
        return small();
    }

    /** Extracts entities of the {@linkplain Builder#labels(List) default labels}. */
    @Override
    public List<NamedEntity> recognize(String text) {
        return recognize(text, defaultLabels);
    }

    @Override
    public List<NamedEntity> recognize(String text, List<String> labels) {
        if (labels.isEmpty()) {
            throw new IllegalArgumentException("At least one label is required");
        }
        List<Word> words = split(text);
        if (words.isEmpty()) {
            return List.of();
        }
        List<long[]> labelTokens = new ArrayList<>(labels.size());
        for (String label : labels) {
            labelTokens.add(tokenizer.encode(label, Integer.MAX_VALUE).inputIds());
        }

        List<Span> candidates = new ArrayList<>();
        if (stride != null && words.size() > maxWords) {
            for (TokenWindows.Window window : TokenWindows.plan(words.size(), maxWords, stride)) {
                for (Span span : candidates(words.subList(window.start(), window.end()), labelTokens)) {
                    int start = span.start() + window.start();
                    if (start >= window.ownedStart() && start < window.ownedEnd()) {
                        candidates.add(new Span(start, span.end() + window.start(), span.label(), span.score()));
                    }
                }
            }
        } else {
            List<Word> kept = truncationGuard.check(words.size(), maxWords, "words") ? words.subList(0, maxWords) : words;
            candidates.addAll(candidates(kept, labelTokens));
        }

        List<NamedEntity> entities = new ArrayList<>();
        for (Span span : greedy(candidates)) {
            int start = words.get(span.start()).start();
            int end = words.get(span.end()).end();
            entities.add(new NamedEntity(text.substring(start, end), labels.get(span.label()), start, end, span.score()));
        }
        return entities;
    }

    @Override
    public void close() {
        session.close();
    }

    record Word(String text, int start, int end) {
    }

    /** A span of words {@code [start, end]} (inclusive) with a label index and probability. */
    private record Span(int start, int end, int label, float score) {
    }

    private static List<Word> split(String text) {
        List<Word> words = new ArrayList<>();
        Matcher matcher = WORDS.matcher(text);
        while (matcher.find()) {
            words.add(new Word(matcher.group(), matcher.start(), matcher.end()));
        }
        return words;
    }

    /** Runs the model on one sequence of words; returns the spans above the threshold, in gliner's enumeration order. */
    private List<Span> candidates(List<Word> words, List<long[]> labelTokens) {
        int numWords = words.size();
        List<Long> ids = new ArrayList<>();
        List<Long> wordsMask = new ArrayList<>();
        for (long id : special.singlePrefix()) {
            ids.add(id);
            wordsMask.add(0L);
        }
        for (long[] label : labelTokens) {
            ids.add(entToken);
            wordsMask.add(0L);
            for (long id : label) {
                ids.add(id);
                wordsMask.add(0L);
            }
        }
        ids.add(sepToken);
        wordsMask.add(0L);
        for (int w = 0; w < numWords; w++) {
            long[] tokens = tokenizer.encode(words.get(w).text(), Integer.MAX_VALUE).inputIds();
            for (int t = 0; t < tokens.length; t++) {
                ids.add(tokens[t]);
                wordsMask.add(t == 0 ? w + 1L : 0L);   // the word's first subtoken carries its 1-based number
            }
        }
        for (long id : special.singleSuffix()) {
            ids.add(id);
            wordsMask.add(0L);
        }

        int length = ids.size();
        long[] inputIds = ids.stream().mapToLong(Long::longValue).toArray();
        long[] attentionMask = new long[length];
        Arrays.fill(attentionMask, 1L);
        long[] mask = wordsMask.stream().mapToLong(Long::longValue).toArray();
        int numSpans = numWords * maxWidth;
        long[] spanIdx = new long[numSpans * 2];
        boolean[] spanMask = new boolean[numSpans];
        for (int s = 0; s < numWords; s++) {
            for (int k = 0; k < maxWidth; k++) {
                int i = s * maxWidth + k;
                spanIdx[2 * i] = s;
                spanIdx[2 * i + 1] = s + k;
                spanMask[i] = s + k < numWords;
            }
        }

        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(inputIds, new long[]{1, length}));
        inputs.put("attention_mask", Tensor.fromLongs(attentionMask, new long[]{1, length}));
        inputs.put("words_mask", Tensor.fromLongs(mask, new long[]{1, length}));
        inputs.put("text_lengths", Tensor.fromLongs(new long[]{numWords}, new long[]{1, 1}));
        inputs.put("span_idx", Tensor.fromLongs(spanIdx, new long[]{1, numSpans, 2}));
        inputs.put("span_mask", Tensor.fromBooleans(spanMask, new long[]{1, numSpans}));
        Tensor logits = session.run(inputs, Set.of("logits")).get("logits");
        long[] shape = logits.shape();   // [1, words, max_width, labels]
        int outWords = (int) shape[1];
        int widths = (int) shape[2];
        int classes = (int) shape[3];
        float[] values = logits.toFloats();

        List<Span> spans = new ArrayList<>();
        for (int s = 0; s < outWords; s++) {
            for (int k = 0; k < widths; k++) {
                for (int c = 0; c < classes; c++) {
                    float score = (float) (1.0 / (1.0 + Math.exp(-values[(s * widths + k) * classes + c])));
                    if (score > threshold && s + k + 1 <= numWords) {
                        spans.add(new Span(s, s + k, c, score));
                    }
                }
            }
        }
        return spans;
    }

    /** gliner's {@code greedy_search} for flat NER: best first, no overlaps, then in text order. */
    private List<Span> greedy(List<Span> candidates) {
        List<Span> byScore = new ArrayList<>(candidates);
        byScore.sort(Comparator.comparingDouble(s -> -s.score()));   // stable: ties keep enumeration order
        List<Span> kept = new ArrayList<>();
        for (Span span : byScore) {
            boolean overlaps = false;
            for (Span other : kept) {
                if (span.start() == other.start() && span.end() == other.end()
                        ? !multiLabel
                        : !(span.start() > other.end() || other.start() > span.end())) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                kept.add(span);
            }
        }
        kept.sort(Comparator.comparingInt(Span::start));
        return kept;
    }

    public static class Builder {
        private ModelSource modelSource;
        private String modelId = SMALL;
        private SessionConfigurer sessionConfigurer;
        private List<String> labels = GENERAL_LABELS;
        private float threshold = 0.5f;
        private boolean multiLabel;
        private TruncationPolicy truncation;
        private Integer stride;

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        /** The entity types {@link #recognize(String)} looks for. Defaults to the preset's labels. */
        public Builder labels(List<String> labels) {
            this.labels = List.copyOf(labels);
            return this;
        }

        /** Spans scoring at or below this are ignored. Defaults to 0.5, as in gliner; the PII model suggests 0.3. */
        public Builder threshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        /** Whether the exact same span may carry several labels. Defaults to {@code false}. */
        public Builder multiLabel(boolean multiLabel) {
            this.multiLabel = multiLabel;
            return this;
        }

        /**
         * What to do with text of more words than the model reads ({@code max_len} in {@code gliner_config.json}).
         * Defaults to {@link TruncationPolicy#TRUNCATE}, which keeps the first words and logs a warning.
         */
        public Builder truncation(TruncationPolicy truncation) {
            this.truncation = truncation;
            return this;
        }

        /**
         * Reads text longer than the model's word limit in windows that share {@code stride} words, instead of
         * truncating it. Off by default.
         */
        public Builder stride(int stride) {
            this.stride = stride;
            return this;
        }

        public GlinerEntityRecognizer build() {
            if (labels.isEmpty()) {
                throw new IllegalArgumentException("At least one default label is required");
            }
            if (threshold < 0f || threshold >= 1f) {
                throw new IllegalArgumentException("threshold must be in [0, 1), got " + threshold);
            }
            ModelSource source = modelSource != null ? modelSource : HuggingFaceModelSource.defaultInstance();
            Path dir = source.resolve(modelId, List.of("model.onnx", "tokenizer.json", "gliner_config.json"));
            Path modelPath = requireFile(dir, "model.onnx");
            Path tokenizerPath = requireFile(dir, "tokenizer.json");
            JsonNode config = readJson(requireFile(dir, "gliner_config.json"));

            String spanMode = config.path("span_mode").asText();
            if (!"markerV0".equals(spanMode)) {
                throw new ModelLoadException("Unsupported GLiNER span_mode '" + spanMode + "'; only markerV0 is supported");
            }
            int maxWidth = config.path("max_width").asInt(12);
            int maxWords = config.path("max_len").asInt(384);
            if (stride != null && (stride < 0 || stride >= maxWords)) {
                throw new IllegalArgumentException("stride must be in [0, " + maxWords + "), got " + stride);
            }
            Map<String, Integer> added = TokenizerJsonParser.parseAddedTokens(tokenizerPath);
            Integer ent = added.get(config.path("ent_token").asText("<<ENT>>"));
            Integer sep = added.get(config.path("sep_token").asText("<<SEP>>"));
            if (ent == null || sep == null) {
                throw new ModelLoadException("tokenizer.json lacks GLiNER's entity or separator token");
            }

            Tokenizer tokenizer = TokenizerJsonParser.parseUnigram(tokenizerPath).build();
            SpecialTokens special = TokenizerJsonParser.parseSpecialTokens(tokenizerPath);
            InferenceSession session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            return new GlinerEntityRecognizer(this, session, tokenizer, special, ent, sep, maxWidth, maxWords);
        }

        private static JsonNode readJson(Path path) {
            try {
                return new ObjectMapper().readTree(path.toFile());
            } catch (IOException e) {
                throw new ModelSourceException("Failed to read " + path + ": " + e.getMessage(), e);
            }
        }

        private static Path requireFile(Path dir, String name) {
            Path file = dir.resolve(name);
            if (!Files.exists(file)) {
                throw new ModelSourceException("Model file not found: " + file);
            }
            return file;
        }
    }
}
