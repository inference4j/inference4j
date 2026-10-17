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
import io.github.inference4j.preprocessing.text.ModelConfig;
import io.github.inference4j.processing.TokenWindows;
import io.github.inference4j.processing.WindowAggregation;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.tokenizer.SpecialTokens;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.TokenizerJsonParser;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Zero-shot text classifier backed by a natural language inference (NLI) model.
 *
 * <p>Each candidate label is turned into a hypothesis with the {@linkplain Builder#hypothesisTemplate(String)
 * hypothesis template} ("This text is about billing."), and the model scores whether the text entails it. This is
 * how HuggingFace's {@code zero-shot-classification} pipeline works, and the scores match it.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * try (var classifier = DebertaZeroShotClassifier.base().build()) {    // ~740 MB (default)
 *     classifier.classify(text, List.of("billing", "outage", "feature request"));
 * }
 * DebertaZeroShotClassifier.large()                                   // ~1.7 GB, more accurate
 * }</pre>
 * <p>Both are Moritz Laurer's {@code deberta-v3-*-zeroshot-v2.0-c} models: MIT-licensed and trained only on
 * commercially-friendly data.
 *
 * <h2>Single-label and multi-label</h2>
 * <p>By default the labels compete: scores are a softmax over the labels' entailment scores and sum to 1. With
 * {@link Builder#multiLabel(boolean) multiLabel(true)}, each label is judged on its own (entailment against
 * contradiction), so several labels, or none, can score high.
 *
 * <h2>Long text</h2>
 * <p>Text that does not fit in {@code maxLength} tokens together with a hypothesis is truncated at its end, as in
 * HuggingFace. With {@link Builder#stride(int)} set, the whole text is scored in overlapping windows instead and the
 * window scores are combined by a {@link WindowAggregation}: {@code mean()} by default for single-label (an overall
 * verdict), {@code max()} for multi-label (a label that applies anywhere applies).
 *
 * @see ZeroShotTextClassifier
 */
public class DebertaZeroShotClassifier implements ZeroShotTextClassifier {

    private static final String BASE = "inference4j/deberta-v3-base-zeroshot-v2.0-c";
    private static final String LARGE = "inference4j/deberta-v3-large-zeroshot-v2.0-c";
    private static final String DEFAULT_TEMPLATE = "This text is about {}.";
    private static final int DEFAULT_MAX_LENGTH = 512;

    private final InferenceSession session;
    private final Tokenizer tokenizer;
    private final SpecialTokens special;
    private final int entailmentIndex;
    private final int contradictionIndex;
    private final long padId;
    private final String template;
    private final boolean multiLabel;
    private final int maxLength;
    private final Integer stride;
    private final WindowAggregation windowAggregation;
    private final boolean needsTokenTypeIds;

    private DebertaZeroShotClassifier(Builder builder, InferenceSession session, Tokenizer tokenizer,
                                      SpecialTokens special, int entailmentIndex, int contradictionIndex, long padId) {
        this.session = session;
        this.tokenizer = tokenizer;
        this.special = special;
        this.entailmentIndex = entailmentIndex;
        this.contradictionIndex = contradictionIndex;
        this.padId = padId;
        this.template = builder.template;
        this.multiLabel = builder.multiLabel;
        this.maxLength = builder.maxLength;
        this.stride = builder.stride;
        this.windowAggregation = builder.windowAggregation != null ? builder.windowAggregation
                : builder.multiLabel ? WindowAggregation.max() : WindowAggregation.mean();
        this.needsTokenTypeIds = session.inputNames().contains("token_type_ids");
    }

    /** DeBERTa-v3 base zero-shot model (~740 MB), the default. */
    public static Builder base() {
        return new Builder().modelId(BASE);
    }

    /** DeBERTa-v3 large zero-shot model (~1.7 GB): more accurate, about three times slower. */
    public static Builder large() {
        return new Builder().modelId(LARGE);
    }

    /** A builder for the default model, {@link #base()}. */
    public static Builder builder() {
        return base();
    }

    @Override
    public List<TextClassification> classify(String text, List<String> candidateLabels) {
        if (candidateLabels.isEmpty()) {
            throw new IllegalArgumentException("At least one candidate label is required");
        }
        long[] premise = tokenizer.encode(text, Integer.MAX_VALUE).inputIds();
        List<long[]> hypotheses = new ArrayList<>(candidateLabels.size());
        int longestHypothesis = 0;
        for (String label : candidateLabels) {
            long[] hypothesis = tokenizer.encode(template.replace("{}", label), Integer.MAX_VALUE).inputIds();
            hypotheses.add(hypothesis);
            longestHypothesis = Math.max(longestHypothesis, hypothesis.length);
        }

        float[] scores;
        int windowSize = maxLength - special.pairCount() - longestHypothesis;
        if (stride != null && premise.length > windowSize) {
            if (windowSize <= stride) {
                throw new IllegalArgumentException("The hypotheses leave " + windowSize
                        + " tokens per window, not more than the stride " + stride);
            }
            List<float[]> windowScores = new ArrayList<>();
            for (TokenWindows.Window window : TokenWindows.plan(premise.length, windowSize, stride)) {
                windowScores.add(score(Arrays.copyOfRange(premise, window.start(), window.end()), hypotheses));
            }
            scores = windowAggregation.aggregate(windowScores);
        } else {
            scores = score(premise, hypotheses);
        }

        List<TextClassification> results = new ArrayList<>(candidateLabels.size());
        for (int i = 0; i < candidateLabels.size(); i++) {
            results.add(new TextClassification(candidateLabels.get(i), i, scores[i]));
        }
        results.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
        return results;
    }

    @Override
    public void close() {
        session.close();
    }

    /** One score per hypothesis for the given premise tokens, all pairs in one padded batch. */
    private float[] score(long[] premise, List<long[]> hypotheses) {
        int count = hypotheses.size();
        List<long[]> pairs = new ArrayList<>(count);
        List<Integer> firstLengths = new ArrayList<>(count);
        int longest = 0;
        for (long[] hypothesis : hypotheses) {
            // Like HuggingFace's only_first truncation: the text gives way, the hypothesis is kept whole
            int budget = Math.max(0, maxLength - special.pairCount() - hypothesis.length);
            long[] first = premise.length > budget ? Arrays.copyOf(premise, budget) : premise;
            long[] pair = special.wrap(first, hypothesis);
            pairs.add(pair);
            firstLengths.add(special.pairPrefix().length + first.length + special.pairMiddle().length);
            longest = Math.max(longest, pair.length);
        }

        long[] inputIds = new long[count * longest];
        long[] attentionMask = new long[count * longest];
        long[] tokenTypeIds = new long[count * longest];
        Arrays.fill(inputIds, padId);
        for (int i = 0; i < count; i++) {
            long[] pair = pairs.get(i);
            System.arraycopy(pair, 0, inputIds, i * longest, pair.length);
            Arrays.fill(attentionMask, i * longest, i * longest + pair.length, 1L);
            Arrays.fill(tokenTypeIds, i * longest + firstLengths.get(i), i * longest + pair.length, 1L);
        }
        long[] shape = {count, longest};
        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(inputIds, shape));
        inputs.put("attention_mask", Tensor.fromLongs(attentionMask, shape));
        if (needsTokenTypeIds) {
            inputs.put("token_type_ids", Tensor.fromLongs(tokenTypeIds, shape));
        }
        Tensor output = session.run(inputs).values().iterator().next();
        int classes = (int) output.shape()[1];
        float[] logits = output.toFloats();

        float[] scores = new float[count];
        if (multiLabel || count == 1) {
            // Each label on its own: softmax over (contradiction, entailment), as HuggingFace does
            for (int i = 0; i < count; i++) {
                double entail = logits[i * classes + entailmentIndex];
                double contradict = logits[i * classes + contradictionIndex];
                scores[i] = (float) (1.0 / (1.0 + Math.exp(contradict - entail)));
            }
        } else {
            // Labels compete: softmax over the labels' entailment logits
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < count; i++) {
                max = Math.max(max, logits[i * classes + entailmentIndex]);
            }
            double sum = 0;
            double[] exp = new double[count];
            for (int i = 0; i < count; i++) {
                exp[i] = Math.exp(logits[i * classes + entailmentIndex] - max);
                sum += exp[i];
            }
            for (int i = 0; i < count; i++) {
                scores[i] = (float) (exp[i] / sum);
            }
        }
        return scores;
    }

    public static class Builder {
        private ModelSource modelSource;
        private String modelId = BASE;
        private SessionConfigurer sessionConfigurer;
        private String template = DEFAULT_TEMPLATE;
        private boolean multiLabel;
        private int maxLength = DEFAULT_MAX_LENGTH;
        private Integer stride;
        private WindowAggregation windowAggregation;

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

        /**
         * How a label becomes a hypothesis; {@code {}} stands for the label. Defaults to
         * {@code "This text is about {}."}. A template that fits your texts ("The customer is asking about {}.")
         * often scores better.
         */
        public Builder hypothesisTemplate(String template) {
            this.template = template;
            return this;
        }

        /**
         * {@code false} (the default): labels compete and scores sum to 1. {@code true}: each label is scored on its
         * own, from 0 to 1, so several labels, or none, can apply.
         */
        public Builder multiLabel(boolean multiLabel) {
            this.multiLabel = multiLabel;
            return this;
        }

        /** Maximum tokens per model run, text, hypothesis and special tokens included. Defaults to 512. */
        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        /**
         * Scores text longer than {@code maxLength} in full, in windows that share {@code stride} tokens, instead
         * of truncating it. Off by default.
         */
        public Builder stride(int stride) {
            this.stride = stride;
            return this;
        }

        /**
         * How the scores of a long text's windows are combined. Defaults to {@link WindowAggregation#mean()} for
         * single-label and {@link WindowAggregation#max()} for multi-label. Requires {@link #stride(int)}.
         */
        public Builder windowAggregation(WindowAggregation windowAggregation) {
            this.windowAggregation = windowAggregation;
            return this;
        }

        public DebertaZeroShotClassifier build() {
            if (!template.contains("{}")) {
                throw new IllegalArgumentException("The hypothesis template must contain {} for the label: " + template);
            }
            if (stride != null && stride < 0) {
                throw new IllegalArgumentException("stride must not be negative, got " + stride);
            }
            if (windowAggregation != null && stride == null) {
                throw new IllegalStateException("windowAggregation requires stride(int)");
            }
            ModelSource source = modelSource != null ? modelSource : HuggingFaceModelSource.defaultInstance();
            Path dir = source.resolve(modelId, List.of("model.onnx", "tokenizer.json", "config.json"));
            Path modelPath = requireFile(dir, "model.onnx");
            Path tokenizerPath = requireFile(dir, "tokenizer.json");
            Path configPath = requireFile(dir, "config.json");

            Tokenizer tokenizer = TokenizerJsonParser.parseUnigram(tokenizerPath).build();
            SpecialTokens special = TokenizerJsonParser.parseSpecialTokens(tokenizerPath);
            ModelConfig config = ModelConfig.fromFile(configPath);
            int entailment = -1;
            for (int i = 0; i < config.numLabels(); i++) {
                if (config.label(i).toLowerCase().startsWith("entail")) {
                    entailment = i;
                }
            }
            if (entailment < 0) {
                throw new ModelLoadException("No entailment label in config.json id2label; is this an NLI model?");
            }
            // HuggingFace's choice: the last class when entailment is the first, otherwise the first
            int contradiction = entailment == 0 ? config.numLabels() - 1 : 0;

            InferenceSession session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            return new DebertaZeroShotClassifier(this, session, tokenizer, special, entailment, contradiction,
                    padTokenId(configPath));
        }

        private static long padTokenId(Path configPath) {
            try {
                JsonNode pad = new ObjectMapper().readTree(configPath.toFile()).path("pad_token_id");
                return pad.isNumber() ? pad.asLong() : 0;
            } catch (IOException e) {
                throw new ModelSourceException("Failed to read config.json: " + e.getMessage(), e);
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
