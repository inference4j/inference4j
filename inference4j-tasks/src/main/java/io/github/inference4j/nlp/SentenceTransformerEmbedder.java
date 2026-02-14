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

import io.github.inference4j.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.ModelSource;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
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

public class SentenceTransformerEmbedder implements TextEmbedder {

    private final InferenceSession session;
    private final Tokenizer tokenizer;
    private final PoolingStrategy poolingStrategy;
    private final int maxLength;

    private SentenceTransformerEmbedder(InferenceSession session, Tokenizer tokenizer,
                                        PoolingStrategy poolingStrategy, int maxLength) {
        this.session = session;
        this.tokenizer = tokenizer;
        this.poolingStrategy = poolingStrategy;
        this.maxLength = maxLength;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public float[] encode(String text) {
        EncodedInput encoded = tokenizer.encode(text, maxLength);

        long[] shape = {1, encoded.inputIds().length};
        Set<String> expectedInputs = session.inputNames();

        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(encoded.inputIds(), shape));
        inputs.put("attention_mask", Tensor.fromLongs(encoded.attentionMask(), shape));
        if (expectedInputs.contains("token_type_ids")) {
            inputs.put("token_type_ids", Tensor.fromLongs(encoded.tokenTypeIds(), shape));
        }

        Map<String, Tensor> outputs = session.run(inputs);

        Tensor outputTensor = outputs.values().iterator().next();
        return applyPooling(outputTensor.toFloats(), outputTensor.shape(),
                encoded.attentionMask(), poolingStrategy);
    }

    @Override
    public List<float[]> encodeBatch(List<String> texts) {
        List<float[]> results = new ArrayList<>(texts.size());
        for (String text : texts) {
            results.add(encode(text));
        }
        return results;
    }

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

    @Override
    public void close() {
        session.close();
    }

    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private Tokenizer tokenizer;
        private PoolingStrategy poolingStrategy = PoolingStrategy.MEAN;
        private int maxLength = 512;

        public Builder session(InferenceSession session) {
            this.session = session;
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

        public Builder poolingStrategy(PoolingStrategy poolingStrategy) {
            this.poolingStrategy = poolingStrategy;
            return this;
        }

        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
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
                Path dir = source.resolve(modelId);
                loadFromDirectory(dir);
            }
            if (tokenizer == null) {
                throw new IllegalStateException("Tokenizer is required");
            }
            return new SentenceTransformerEmbedder(session, tokenizer, poolingStrategy, maxLength);
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

            this.session = InferenceSession.create(modelPath);
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
