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

package io.github.inference4j.text;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.MathOps;
import io.github.inference4j.ModelSource;
import io.github.inference4j.OutputOperator;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.WordPieceTokenizer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Text classification model for DistilBERT/BERT architectures.
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
 * try (DistilBertClassifier model = DistilBertClassifier.fromPretrained("models/distilbert-sst2")) {
 *     List<TextClassification> results = model.classify("This movie was great!");
 *     System.out.println(results.get(0).label()); // "POSITIVE"
 * }
 * }</pre>
 *
 * <h2>Custom configuration</h2>
 * <pre>{@code
 * try (DistilBertClassifier model = DistilBertClassifier.builder()
 *         .session(InferenceSession.create(modelPath))
 *         .tokenizer(WordPieceTokenizer.fromVocabFile(vocabPath))
 *         .config(ModelConfig.fromFile(configPath))
 *         .build()) {
 *     List<TextClassification> results = model.classify("Great product!", 2);
 * }
 * }</pre>
 *
 * @see TextClassificationModel
 * @see TextClassification
 */
public class DistilBertClassifier implements TextClassificationModel {

    private static final int DEFAULT_MAX_LENGTH = 512;

    private final InferenceSession session;
    private final Tokenizer tokenizer;
    private final ModelConfig config;
    private final OutputOperator outputOperator;
    private final int maxLength;

    private DistilBertClassifier(InferenceSession session, Tokenizer tokenizer,
                                  ModelConfig config, OutputOperator outputOperator,
                                  int maxLength) {
        this.session = session;
        this.tokenizer = tokenizer;
        this.config = config;
        this.outputOperator = outputOperator;
        this.maxLength = maxLength;
    }

    public static DistilBertClassifier fromPretrained(String modelPath) {
        Path dir = Path.of(modelPath);
        return fromModelDirectory(dir);
    }

    public static DistilBertClassifier fromPretrained(String modelId, ModelSource source) {
        Path dir = source.resolve(modelId);
        return fromModelDirectory(dir);
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
        float[] logits = outputTensor.toFloats();

        return postProcess(logits, config, topK, outputOperator);
    }

    @Override
    public void close() {
        session.close();
    }

    /**
     * Post-processes raw logits into text classification results.
     *
     * <p>Package-visible for unit testing without an ONNX session.
     */
    static List<TextClassification> postProcess(float[] logits, ModelConfig config,
                                                 int topK, OutputOperator outputOperator) {
        float[] probabilities = outputOperator.apply(logits);
        int[] topIndices = MathOps.topK(probabilities, topK);

        List<TextClassification> results = new ArrayList<>(topIndices.length);
        for (int idx : topIndices) {
            results.add(new TextClassification(config.label(idx), idx, probabilities[idx]));
        }
        return results;
    }

    private static DistilBertClassifier fromModelDirectory(Path dir) {
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

        InferenceSession session = InferenceSession.create(modelPath);
        try {
            Tokenizer tokenizer = WordPieceTokenizer.fromVocabFile(vocabPath);
            ModelConfig config = ModelConfig.fromFile(configPath);
            OutputOperator operator = config.isMultiLabel()
                    ? OutputOperator.sigmoid()
                    : OutputOperator.softmax();
            return new DistilBertClassifier(session, tokenizer, config, operator, DEFAULT_MAX_LENGTH);
        } catch (Exception e) {
            session.close();
            throw e;
        }
    }

    public static class Builder {
        private InferenceSession session;
        private Tokenizer tokenizer;
        private ModelConfig config;
        private OutputOperator outputOperator;
        private int maxLength = DEFAULT_MAX_LENGTH;

        public Builder session(InferenceSession session) {
            this.session = session;
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

        public DistilBertClassifier build() {
            if (session == null) {
                throw new IllegalStateException("InferenceSession is required");
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
            return new DistilBertClassifier(session, tokenizer, config, outputOperator, maxLength);
        }
    }
}
