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
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.model.ModelSource;
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

    private SentenceTransformerEmbedder(InferenceSession session, Tokenizer tokenizer,
                                        PoolingStrategy poolingStrategy, boolean normalize,
                                        String textPrefix, int maxLength) {
        super(session,
                createPreprocessor(tokenizer, maxLength, session.inputNames(), textPrefix),
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
        return run(text);
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

    private static io.github.inference4j.processing.Preprocessor<String, Map<String, Tensor>> createPreprocessor(
            Tokenizer tokenizer, int maxLength, Set<String> expectedInputs, String textPrefix) {
        return text -> {
            String input = textPrefix != null ? textPrefix + text : text;
            EncodedInput encoded = tokenizer.encode(input, maxLength);
            long[] shape = {1, encoded.inputIds().length};
            Map<String, Tensor> inputs = new LinkedHashMap<>();
            inputs.put("input_ids", Tensor.fromLongs(encoded.inputIds(), shape));
            inputs.put("attention_mask", Tensor.fromLongs(encoded.attentionMask(), shape));
            if (expectedInputs.contains("token_type_ids")) {
                inputs.put("token_type_ids", Tensor.fromLongs(encoded.tokenTypeIds(), shape));
            }
            return inputs;
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
         * Builds and returns a new {@link SentenceTransformerEmbedder} instance.
         *
         * @return a configured embedder ready for use
         * @throws IllegalStateException if {@code modelId} is not set
         * @throws ModelSourceException if model files cannot be found or loaded
         */
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
            return new SentenceTransformerEmbedder(session, tokenizer, poolingStrategy,
                    normalize, textPrefix, maxLength);
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
