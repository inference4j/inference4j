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
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.TokenWindows;
import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.WordPieceTokenizer;

import io.github.inference4j.processing.MathOps;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Text embedding model based on the Sentence Transformers architecture. Encodes text into
 * fixed-dimensional dense vectors suitable for semantic search, clustering, and similarity tasks.
 *
 * <p>Supports multiple pooling strategies ({@link PoolingStrategy#CLS CLS}, {@link PoolingStrategy#MEAN MEAN},
 * {@link PoolingStrategy#MAX MAX}), optional L2 normalization for cosine similarity, and text prefixes
 * required by some model families (E5, Nomic).
 *
 * <p>Compatible models include all-MiniLM, all-mpnet, BGE, GTE, and E5.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * try (var embedder = SentenceTransformerEmbedder.builder()
 *         .modelId("inference4j/all-MiniLM-L6-v2")
 *         .build()) {
 *     float[] embedding = embedder.encode("Hello, world!");
 * }
 * }</pre>
 *
 * @see TextEmbedder
 * @see PoolingStrategy
 */
public class SentenceTransformerEmbedder
        extends AbstractInferenceTask<String, float[]>
        implements TextEmbedder {

    private final Tokenizer tokenizer;
    private final PoolingStrategy poolingStrategy;
    private final boolean normalize;
    private final String textPrefix;
    private final int maxLength;
    private final Integer stride;
    private final long[] prefixIds;

    private SentenceTransformerEmbedder(InferenceSession session, Tokenizer tokenizer,
                                        PoolingStrategy poolingStrategy, boolean normalize,
                                        String textPrefix, int maxLength,
                                        TruncationGuard truncationGuard, Integer stride,
                                        long[] prefixIds) {
        super(session,
                createPreprocessor(tokenizer, maxLength, session.inputNames(), textPrefix, truncationGuard),
                ctx -> {
                    Tensor outputTensor = ctx.outputs().values().iterator().next();
                    Tensor attentionMaskTensor = ctx.preprocessed().get("attention_mask");
                    long[] attentionMask = attentionMaskTensor.toLongs();
                    float[] result = applyPooling(outputTensor.toFloats(), outputTensor.shape(),
                            attentionMask, poolingStrategy);
                    if (normalize) {
                        result = MathOps.l2Normalize(result);
                    }
                    return result;
                });
        this.tokenizer = tokenizer;
        this.poolingStrategy = poolingStrategy;
        this.normalize = normalize;
        this.textPrefix = textPrefix;
        this.maxLength = maxLength;
        this.stride = stride;
        this.prefixIds = prefixIds;
    }

    /**
     * Creates a new builder for configuring a {@code SentenceTransformerEmbedder}.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Encodes the given text into a dense vector embedding.
     *
     * @param text the input text to encode
     * @return a float array representing the embedding vector
     */
    @Override
    public float[] encode(String text) {
        if (stride == null) {
            return run(text);
        }
        String input = textPrefix != null ? textPrefix + text : text;
        long[] ids = tokenizer.encode(input, Integer.MAX_VALUE).inputIds();
        if (ids.length <= maxLength) {
            return run(text);
        }
        return encodeInWindows(ids);
    }

    /**
     * Embeds each window of a long input and averages the window embeddings, weighting each by the
     * tokens it owns, so every token counts once. The text prefix is repeated in every window.
     */
    private float[] encodeInWindows(long[] ids) {
        int contentStart = 1 + prefixIds.length;                    // after [CLS] and the prefix
        int contentCount = ids.length - contentStart - 1;            // before [SEP]
        int windowSize = maxLength - 2 - prefixIds.length;

        List<float[]> embeddings = new ArrayList<>();
        List<Integer> weights = new ArrayList<>();
        for (TokenWindows.Window window : TokenWindows.plan(contentCount, windowSize, stride)) {
            int windowTokens = window.end() - window.start();
            long[] windowIds = new long[windowTokens + prefixIds.length + 2];
            windowIds[0] = ids[0];
            System.arraycopy(prefixIds, 0, windowIds, 1, prefixIds.length);
            System.arraycopy(ids, contentStart + window.start(), windowIds, 1 + prefixIds.length, windowTokens);
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
            Tensor output = session.run(inputs).values().iterator().next();
            embeddings.add(applyPooling(output.toFloats(), output.shape(), attentionMask, poolingStrategy));
            weights.add(window.ownedEnd() - window.ownedStart());
        }

        float[] average = weightedAverage(embeddings, weights);
        return normalize ? MathOps.l2Normalize(average) : average;
    }

    static float[] weightedAverage(List<float[]> vectors, List<Integer> weights) {
        float[] result = new float[vectors.get(0).length];
        long total = 0;
        for (int i = 0; i < vectors.size(); i++) {
            float[] vector = vectors.get(i);
            int weight = weights.get(i);
            for (int d = 0; d < result.length; d++) {
                result[d] += vector[d] * weight;
            }
            total += weight;
        }
        for (int d = 0; d < result.length; d++) {
            result[d] /= total;
        }
        return result;
    }

    /**
     * Encodes multiple texts into embedding vectors. Each text is encoded independently.
     *
     * @param texts the input texts to encode
     * @return a list of float arrays, one embedding per input text
     */
    @Override
    public List<float[]> encodeBatch(List<String> texts) {
        List<float[]> results = new ArrayList<>(texts.size());
        for (String text : texts) {
            results.add(encode(text));
        }
        return results;
    }

    /**
     * Applies the given pooling strategy to the model's token-level output, producing a single
     * fixed-size embedding vector.
     *
     * @param flatOutput   the flat token-level output from the model, shape {@code [1, seqLen, hiddenSize]}
     * @param shape        the tensor shape {@code [batch, seqLen, hiddenSize]}
     * @param attentionMask the attention mask indicating real tokens (1) vs padding (0)
     * @param strategy     the pooling strategy to apply
     * @return the pooled embedding vector of size {@code hiddenSize}
     */
    static float[] applyPooling(float[] flatOutput, long[] shape,
                                long[] attentionMask, PoolingStrategy strategy) {
        int seqLen = (int) shape[1];
        int hiddenSize = (int) shape[2];

        return switch (strategy) {
            case CLS -> {
                float[] result = new float[hiddenSize];
                System.arraycopy(flatOutput, 0, result, 0, hiddenSize);
                yield result;
            }
            case MEAN -> {
                float[] result = new float[hiddenSize];
                int count = 0;
                for (int t = 0; t < seqLen; t++) {
                    if (attentionMask[t] == 1) {
                        for (int h = 0; h < hiddenSize; h++) {
                            result[h] += flatOutput[t * hiddenSize + h];
                        }
                        count++;
                    }
                }
                if (count > 0) {
                    for (int h = 0; h < hiddenSize; h++) {
                        result[h] /= count;
                    }
                }
                yield result;
            }
            case MAX -> {
                float[] result = new float[hiddenSize];
                Arrays.fill(result, -Float.MAX_VALUE);
                boolean anyValid = false;
                for (int t = 0; t < seqLen; t++) {
                    if (attentionMask[t] == 1) {
                        anyValid = true;
                        for (int h = 0; h < hiddenSize; h++) {
                            result[h] = Math.max(result[h], flatOutput[t * hiddenSize + h]);
                        }
                    }
                }
                if (!anyValid) {
                    Arrays.fill(result, 0f);
                }
                yield result;
            }
        };
    }

    private static io.github.inference4j.processing.Preprocessor<String, PreprocessResult> createPreprocessor(
            Tokenizer tokenizer, int maxLength, Set<String> expectedInputs, String textPrefix,
            TruncationGuard truncationGuard) {
        return text -> {
            String input = textPrefix != null ? textPrefix + text : text;
            EncodedInput encoded = tokenizer.encode(input, maxLength);
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

    /**
     * Builder for configuring and creating a {@link SentenceTransformerEmbedder}.
     *
     * <p>At minimum, a {@code modelId} must be provided. The builder automatically downloads
     * the model and loads the WordPiece tokenizer from the resolved directory.
     *
     * <pre>{@code
     * var embedder = SentenceTransformerEmbedder.builder()
     *         .modelId("inference4j/bge-base-en-v1.5")
     *         .poolingStrategy(PoolingStrategy.CLS)
     *         .normalize()
     *         .build();
     * }</pre>
     */
    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private Tokenizer tokenizer;
        private PoolingStrategy poolingStrategy = PoolingStrategy.MEAN;
        private boolean normalize = false;
        private String textPrefix;
        private int maxLength = 512;
        private TruncationPolicy truncation;
        private Integer stride;

        Builder session(InferenceSession session) {
            this.session = session;
            return this;
        }

        /**
         * Configures the ONNX Runtime session options (e.g., hardware acceleration).
         *
         * @param sessionConfigurer the session configuration callback
         * @return this builder
         */
        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        /**
         * Sets a custom model source for resolving model files.
         * Defaults to {@link HuggingFaceModelSource} if not specified.
         *
         * @param modelSource the model source to use
         * @return this builder
         */
        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        /**
         * Sets the HuggingFace model ID to download and use.
         *
         * @param modelId the model ID (e.g., {@code "inference4j/all-MiniLM-L6-v2"})
         * @return this builder
         */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * Sets a custom tokenizer. If not specified, a {@link WordPieceTokenizer} is
         * automatically loaded from the model directory's {@code vocab.txt}.
         *
         * @param tokenizer the tokenizer to use
         * @return this builder
         */
        public Builder tokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        /**
         * Sets the pooling strategy for converting token-level outputs into a single embedding.
         * Defaults to {@link PoolingStrategy#MEAN}.
         *
         * @param poolingStrategy the pooling strategy
         * @return this builder
         * @see PoolingStrategy
         */
        public Builder poolingStrategy(PoolingStrategy poolingStrategy) {
            this.poolingStrategy = poolingStrategy;
            return this;
        }

        /**
         * Enables L2 normalization of the output embeddings. Recommended when using
         * cosine similarity for comparison (e.g., BGE, GTE, E5 models).
         */
        public Builder normalize() {
            this.normalize = true;
            return this;
        }

        /**
         * Sets a text prefix to prepend to all inputs before encoding.
         * Required by some models: E5 uses {@code "query: "} or {@code "passage: "},
         * Nomic uses {@code "search_query: "} or {@code "search_document: "}.
         *
         * @param textPrefix the prefix string to prepend (e.g., "query: ")
         */
        public Builder textPrefix(String textPrefix) {
            this.textPrefix = textPrefix;
            return this;
        }

        /**
         * Sets the maximum token sequence length. Inputs longer than this are truncated.
         * Defaults to 512.
         *
         * @param maxLength the maximum sequence length
         * @return this builder
         */
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
         * Builds and returns a new {@link SentenceTransformerEmbedder} instance.
         *
         * @return a configured embedder ready for use
         * @throws IllegalStateException if {@code modelId} is not set
         * @throws ModelSourceException if model files cannot be found or loaded
         */
        /**
         * Embeds input longer than {@code maxLength} in full: the text is split into windows that
         * share {@code stride} tokens, each window is embedded, and the result is the average of the
         * window embeddings weighted by the tokens each window covers. {@code 0} gives consecutive,
         * non-overlapping chunks. Off by default. When set, input is never truncated, so
         * {@link #truncation(TruncationPolicy)} does not apply.
         *
         * <p>Splitting documents into chunks before embedding usually retrieves better than one
         * averaged vector; use this when a single vector per document is required.
         */
        public Builder stride(int stride) {
            this.stride = stride;
            return this;
        }

        public SentenceTransformerEmbedder build() {
            if (session == null) {
                if (modelId == null) {
                    throw new IllegalStateException(
                            "modelId is required (e.g., \"inference4j/all-MiniLM-L6-v2\")");
                }
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                Path dir = source.resolve(modelId, List.of("model.onnx", "vocab.txt"));
                loadFromDirectory(dir);
            }
            if (tokenizer == null) {
                throw new IllegalStateException("Tokenizer is required");
            }
            long[] prefixIds = new long[0];
            if (stride != null) {
                if (textPrefix != null) {
                    long[] encoded = tokenizer.encode(textPrefix, Integer.MAX_VALUE).inputIds();
                    prefixIds = Arrays.copyOfRange(encoded, 1, encoded.length - 1);   // without [CLS]/[SEP]
                }
                int windowSize = maxLength - 2 - prefixIds.length;
                if (stride < 0 || stride >= windowSize) {
                    throw new IllegalArgumentException("stride must be in [0, " + windowSize + ") for maxLength "
                            + maxLength + (prefixIds.length > 0 ? " and the text prefix" : "") + ", got " + stride);
                }
            }
            return new SentenceTransformerEmbedder(session, tokenizer, poolingStrategy,
                    normalize, textPrefix, maxLength,
                    new TruncationGuard("SentenceTransformerEmbedder", truncation), stride, prefixIds);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            Path vocabPath = dir.resolve("vocab.txt");

            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }
            if (!Files.exists(vocabPath)) {
                throw new ModelSourceException("Vocabulary file not found: " + vocabPath);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            try {
                if (this.tokenizer == null) {
                    this.tokenizer = WordPieceTokenizer.fromVocabFile(vocabPath);
                }
            } catch (Exception e) {
                this.session.close();
                this.session = null;
                throw e;
            }
        }
    }
}
