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
import io.github.inference4j.processing.Preprocessor;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.WordPieceTokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Cross-encoder search reranker based on MiniLM architecture.
 *
 * <h2>Target model</h2>
 * <p>Designed for <a href="https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2">
 * cross-encoder/ms-marco-MiniLM-L-6-v2</a>, a compact and fast cross-encoder trained
 * on MS MARCO passage ranking data.
 *
 * <p>The model directory should contain:
 * <ul>
 *   <li>{@code model.onnx} — the ONNX model file</li>
 *   <li>{@code vocab.txt} — WordPiece vocabulary</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (MiniLMSearchReranker reranker = MiniLMSearchReranker.builder().build()) {
 *     float score = reranker.score("What is Java?", "Java is a programming language.");
 *
 *     // Re-rank multiple documents
 *     float[] scores = reranker.scoreBatch("What is Java?", List.of(
 *         "Java is a programming language.",
 *         "Python is a programming language.",
 *         "The weather is nice today."
 *     ));
 * }
 * }</pre>
 *
 * @see SearchReranker
 */
public class MiniLMSearchReranker
        extends AbstractInferenceTask<QueryDocumentPair, Float>
        implements io.github.inference4j.nlp.SearchReranker {

    private static final String DEFAULT_MODEL_ID = "inference4j/ms-marco-MiniLM-L-6-v2";

    private static final int DEFAULT_MAX_LENGTH = 512;

    private MiniLMSearchReranker(InferenceSession session, Tokenizer tokenizer, int maxLength) {
        super(session,
                createPreprocessor(tokenizer, maxLength, session.inputNames()),
                ctx -> {
                    Tensor outputTensor = ctx.outputs().values().iterator().next();
                    float[] logits = outputTensor.toFloats();
                    return toScore(logits[0]);
                });
    }

    public static Builder builder() {
        return new Builder();
    }

    static float toScore(float logit) {
        return (float) (1.0 / (1.0 + Math.exp(-logit)));
    }

    private static Preprocessor<QueryDocumentPair, Map<String, Tensor>> createPreprocessor(
            Tokenizer tokenizer, int maxLength, Set<String> expectedInputs) {
        return pair -> {
            EncodedInput encoded = tokenizer.encode(pair.query(), pair.document(), maxLength);
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

    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private Tokenizer tokenizer;
        private int maxLength = DEFAULT_MAX_LENGTH;

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

        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        public MiniLMSearchReranker build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id, List.of("model.onnx", "vocab.txt"));
                loadFromDirectory(dir);
            }
            if (tokenizer == null) {
                throw new IllegalStateException("Tokenizer is required");
            }
            return new MiniLMSearchReranker(session, tokenizer, maxLength);
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
