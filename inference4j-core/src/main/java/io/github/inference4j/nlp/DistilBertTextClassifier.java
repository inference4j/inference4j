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

import io.github.inference4j.AbstractInferenceTask;
import io.github.inference4j.PreprocessResult;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.TokenWindows;
import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.processing.OutputOperator;
import io.github.inference4j.processing.WindowAggregation;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.preprocessing.text.ModelConfig;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.WordPieceTokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * DistilBERT text classifier.
 *
 * <h2>Target model</h2>
 * <p>Designed for <a href="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english">
 * distilbert-base-uncased-finetuned-sst-2-english</a> (2-class sentiment analysis).
 * Works with any BERT-family model exported to ONNX that uses WordPiece tokenization.
 *
 * <p>The model directory should contain:
 * <ul>
 *   <li>{@code model.onnx} — the ONNX model file</li>
 *   <li>{@code vocab.txt} — WordPiece vocabulary</li>
 *   <li>{@code config.json} — HuggingFace config with {@code id2label} and optional {@code problem_type}</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (DistilBertTextClassifier classifier = DistilBertTextClassifier.builder().build()) {
 *     List<TextClassification> results = classifier.classify("This movie was great!");
 *     System.out.println(results.get(0).label()); // "POSITIVE"
 * }
 * }</pre>
 *
 * <h2>Custom configuration</h2>
 * <pre>{@code
 * try (DistilBertTextClassifier classifier = DistilBertTextClassifier.builder()
 *         .modelId("my-org/my-distilbert")
 *         .modelSource(ModelSource.fromPath(localDir))
 *         .sessionOptions(opts -> opts.addCUDA(0))
 *         .tokenizer(WordPieceTokenizer.fromVocabFile(vocabPath))
 *         .config(ModelConfig.fromFile(configPath))
 *         .build()) {
 *     List<TextClassification> results = classifier.classify("Great product!", 2);
 * }
 * }</pre>
 *
 * <h2>Long texts</h2>
 * <p>Text longer than {@code maxLength} tokens is truncated by default. With
 * {@link Builder#stride(int)} set, the whole text is classified in overlapping windows whose
 * scores are combined by a {@link WindowAggregation}, the mean by default:
 * <pre>{@code
 * try (DistilBertTextClassifier classifier = DistilBertTextClassifier.builder()
 *         .stride(128)
 *         .windowAggregation(WindowAggregation.max())
 *         .build()) {
 *     List<TextClassification> results = classifier.classify(longReview);
 * }
 * }</pre>
 *
 * @see TextClassifier
 * @see TextClassification
 */
public class DistilBertTextClassifier
        extends AbstractInferenceTask<String, List<TextClassification>>
        implements TextClassifier {

    private static final String DEFAULT_MODEL_ID = "inference4j/distilbert-base-uncased-finetuned-sst-2-english";
    private static final int DEFAULT_MAX_LENGTH = 512;

    private final Tokenizer tokenizer;
    private final ModelConfig config;
    private final OutputOperator outputOperator;
    private final int maxLength;
    private final Integer stride;
    private final WindowAggregation windowAggregation;

    private DistilBertTextClassifier(InferenceSession session, Tokenizer tokenizer,
                                     ModelConfig config, OutputOperator outputOperator,
                                     int maxLength, TruncationGuard truncationGuard,
                                     Integer stride, WindowAggregation windowAggregation) {
        super(session,
                createPreprocessor(tokenizer, maxLength, session.inputNames(), truncationGuard),
                ctx -> {
                    Tensor outputTensor = ctx.outputs().values().iterator().next();
                    float[] logits = outputTensor.toFloats();
                    return postProcess(logits, config, config.numLabels(), outputOperator);
                });
        this.tokenizer = tokenizer;
        this.config = config;
        this.outputOperator = outputOperator;
        this.maxLength = maxLength;
        this.stride = stride;
        this.windowAggregation = windowAggregation;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<TextClassification> classify(String text) {
        return classify(text, config.numLabels());
    }

    @Override
    public List<TextClassification> classify(String text, int topK) {
        if (stride != null) {
            long[] ids = tokenizer.encode(text, Integer.MAX_VALUE).inputIds();
            if (ids.length > maxLength) {
                return topLabels(windowAggregation.aggregate(scoreWindows(ids)), config, topK);
            }
        }
        PreprocessResult result = preprocessor.process(text);
        Map<String, Tensor> outputs = session.run(result.tensors());
        Tensor outputTensor = outputs.values().iterator().next();
        float[] logits = outputTensor.toFloats();
        return postProcess(logits, config, topK, outputOperator);
    }

    /** Scores every window of a long input: one array of label probabilities per window. */
    private List<float[]> scoreWindows(long[] ids) {
        int contentCount = ids.length - 2;                          // without [CLS] and [SEP]
        List<float[]> windowScores = new ArrayList<>();
        for (TokenWindows.Window window : TokenWindows.plan(contentCount, maxLength - 2, stride)) {
            int windowTokens = window.end() - window.start();
            long[] windowIds = new long[windowTokens + 2];
            windowIds[0] = ids[0];
            System.arraycopy(ids, 1 + window.start(), windowIds, 1, windowTokens);
            windowIds[windowIds.length - 1] = ids[ids.length - 1];

            long[] attentionMask = new long[windowIds.length];
            Arrays.fill(attentionMask, 1L);
            long[] shape = {1, windowIds.length};
            Map<String, Tensor> inputs = new LinkedHashMap<>();
            inputs.put("input_ids", Tensor.fromLongs(windowIds, shape));
            inputs.put("attention_mask", Tensor.fromLongs(attentionMask, shape));
            if (session.inputNames().contains("token_type_ids")) {
                inputs.put("token_type_ids", Tensor.fromLongs(new long[windowIds.length], shape));
            }
            float[] logits = session.run(inputs).values().iterator().next().toFloats();
            windowScores.add(outputOperator.apply(logits));
        }
        return windowScores;
    }

    static List<TextClassification> postProcess(float[] logits, ModelConfig config,
                                                 int topK, OutputOperator outputOperator) {
        return topLabels(outputOperator.apply(logits), config, topK);
    }

    private static List<TextClassification> topLabels(float[] probabilities, ModelConfig config, int topK) {
        int[] topIndices = MathOps.topK(probabilities, topK);

        List<TextClassification> results = new ArrayList<>(topIndices.length);
        for (int idx : topIndices) {
            results.add(new TextClassification(config.label(idx), idx, probabilities[idx]));
        }
        return results;
    }

    private static io.github.inference4j.processing.Preprocessor<String, PreprocessResult> createPreprocessor(
            Tokenizer tokenizer, int maxLength, Set<String> expectedInputs,
            TruncationGuard truncationGuard) {
        return text -> {
            EncodedInput encoded = tokenizer.encode(text, maxLength);
            truncationGuard.check(encoded, maxLength);
            long[] shape = {1, encoded.inputIds().length};
            Map<String, Tensor> inputs = new LinkedHashMap<>();
            inputs.put("input_ids", Tensor.fromLongs(encoded.inputIds(), shape));
            inputs.put("attention_mask", Tensor.fromLongs(encoded.attentionMask(), shape));
            if (expectedInputs.contains("token_type_ids")) {
                inputs.put("token_type_ids", Tensor.fromLongs(encoded.tokenTypeIds(), shape));
            }
            return PreprocessResult.of(inputs);
        };
    }

    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private Tokenizer tokenizer;
        private ModelConfig config;
        private OutputOperator outputOperator;
        private int maxLength = DEFAULT_MAX_LENGTH;
        private TruncationPolicy truncation;
        private Integer stride;
        private WindowAggregation windowAggregation;

        Builder session(InferenceSession session) {
            this.session = session;
            return this;
        }

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder tokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder config(ModelConfig config) {
            this.config = config;
            return this;
        }

        public Builder outputOperator(OutputOperator outputOperator) {
            this.outputOperator = outputOperator;
            return this;
        }

        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        /**
         * What to do with input longer than {@code maxLength} tokens. Defaults to
         * {@link TruncationPolicy#TRUNCATE}, which keeps the first tokens and logs a warning.
         */
        public Builder truncation(TruncationPolicy truncation) {
            this.truncation = truncation;
            return this;
        }

        /**
         * Classifies input longer than {@code maxLength} in full: the text is split into windows that
         * share {@code stride} tokens, each window is classified, and the window scores are combined
         * by the {@link #windowAggregation(WindowAggregation) window aggregation}. {@code 0} gives
         * consecutive, non-overlapping windows. Off by default. When set, input is never truncated,
         * so {@link #truncation(TruncationPolicy)} does not apply.
         */
        public Builder stride(int stride) {
            this.stride = stride;
            return this;
        }

        /**
         * How the scores of a long input's windows are combined into one score per label. Defaults to
         * {@link WindowAggregation#mean()}. Requires {@link #stride(int)}.
         */
        public Builder windowAggregation(WindowAggregation windowAggregation) {
            this.windowAggregation = windowAggregation;
            return this;
        }

        public DistilBertTextClassifier build() {
            if (stride != null && (stride < 0 || stride >= maxLength - 2)) {
                throw new IllegalArgumentException(
                        "stride must be in [0, " + (maxLength - 2) + ") for maxLength " + maxLength + ", got " + stride);
            }
            if (windowAggregation != null && stride == null) {
                throw new IllegalStateException("windowAggregation requires stride(int)");
            }
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id, List.of("model.onnx", "vocab.txt", "config.json"));
                loadFromDirectory(dir);
            }
            if (tokenizer == null) {
                throw new IllegalStateException("Tokenizer is required");
            }
            if (config == null) {
                throw new IllegalStateException("ModelConfig is required");
            }
            if (outputOperator == null) {
                outputOperator = config.isMultiLabel()
                        ? OutputOperator.sigmoid()
                        : OutputOperator.softmax();
            }
            return new DistilBertTextClassifier(session, tokenizer, config, outputOperator, maxLength,
                    new TruncationGuard("DistilBertTextClassifier", truncation), stride,
                    windowAggregation != null ? windowAggregation : WindowAggregation.mean());
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            Path vocabPath = dir.resolve("vocab.txt");
            Path configPath = dir.resolve("config.json");

            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }
            if (!Files.exists(vocabPath)) {
                throw new ModelSourceException("Vocabulary file not found: " + vocabPath);
            }
            if (!Files.exists(configPath)) {
                throw new ModelSourceException("Config file not found: " + configPath);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            try {
                if (this.tokenizer == null) {
                    this.tokenizer = WordPieceTokenizer.fromVocabFile(vocabPath);
                }
                if (this.config == null) {
                    this.config = ModelConfig.fromFile(configPath);
                }
            } catch (Exception e) {
                this.session.close();
                this.session = null;
                throw e;
            }
        }
    }
}
