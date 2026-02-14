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
import io.github.inference4j.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.MathOps;
import io.github.inference4j.ModelSource;
import io.github.inference4j.OutputOperator;
import io.github.inference4j.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.text.ModelConfig;
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

    private DistilBertTextClassifier(InferenceSession session, Tokenizer tokenizer,
                                     ModelConfig config, OutputOperator outputOperator,
                                     int maxLength) {
        super(session,
                createPreprocessor(tokenizer, maxLength, session.inputNames()),
                ctx -> {
                    Tensor outputTensor = ctx.outputs().values().iterator().next();
                    float[] logits = outputTensor.toFloats();
                    return postProcess(logits, config, config.numLabels(), outputOperator);
                });
        this.tokenizer = tokenizer;
        this.config = config;
        this.outputOperator = outputOperator;
        this.maxLength = maxLength;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<TextClassification> classify(String text) {
        return run(text);
    }

    @Override
    public List<TextClassification> classify(String text, int topK) {
        Map<String, Tensor> inputs = preprocessor.process(text);
        Map<String, Tensor> outputs = session.run(inputs);
        Tensor outputTensor = outputs.values().iterator().next();
        float[] logits = outputTensor.toFloats();
        return postProcess(logits, config, topK, outputOperator);
    }

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

    private static io.github.inference4j.Preprocessor<String, Map<String, Tensor>> createPreprocessor(
            Tokenizer tokenizer, int maxLength, Set<String> expectedInputs) {
        return text -> {
            EncodedInput encoded = tokenizer.encode(text, maxLength);
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
        private ModelConfig config;
        private OutputOperator outputOperator;
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

        public DistilBertTextClassifier build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id);
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
            return new DistilBertTextClassifier(session, tokenizer, config, outputOperator, maxLength);
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
