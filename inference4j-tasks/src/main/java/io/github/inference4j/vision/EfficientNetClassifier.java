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

package io.github.inference4j.vision;

import io.github.inference4j.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.ModelSource;
import io.github.inference4j.OutputOperator;
import io.github.inference4j.Preprocessor;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.image.Labels;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Image classifier for EfficientNet architectures.
 *
 * <h2>Tested model</h2>
 * <p>This wrapper is tested against
 * <a href="https://huggingface.co/onnx/EfficientNet-Lite4">EfficientNet-Lite4</a>
 * (ONNX format, TensorFlow origin). That model has <strong>softmax built into the
 * graph</strong>, so this wrapper uses {@link OutputOperator#identity()} by default —
 * the output values are already probabilities.
 *
 * <p><strong>Other EfficientNet exports may differ.</strong> Some ONNX exports (e.g.,
 * from PyTorch's {@code timm} library) output raw logits without a final activation.
 * Using this wrapper with such a model would pass raw logits as confidence scores.
 * If your model outputs logits, override with
 * {@code .outputOperator(OutputOperator.softmax())} via the builder.
 *
 * <h2>Preprocessing</h2>
 * <ul>
 *   <li>Normalization: mean {@code [127/255, 127/255, 127/255]}, std {@code [128/255, 128/255, 128/255]}</li>
 *   <li>Input layout: auto-detected from model (typically NHWC for TF-origin models)</li>
 *   <li>Input size: auto-detected from model (B0=224, B1=240, B2=260, B3=300, B4=380, Lite4=280)</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (EfficientNetClassifier classifier = EfficientNetClassifier.builder().build()) {
 *     List<Classification> results = classifier.classify(Path.of("cat.jpg"));
 * }
 * }</pre>
 *
 * <h2>Larger variants via builder</h2>
 * <pre>{@code
 * try (EfficientNetClassifier classifier = EfficientNetClassifier.builder()
 *         .modelId("my-org/my-efficientnet")
 *         .modelSource(ModelSource.fromPath(localDir))
 *         .sessionOptions(opts -> opts.addCUDA(0))
 *         .labels(Labels.fromFile(Path.of("my-labels.txt")))
 *         .outputOperator(OutputOperator.softmax()) // if model outputs raw logits
 *         .defaultTopK(10)
 *         .build()) {
 *     List<Classification> results = classifier.classify(image, 3);
 * }
 * }</pre>
 *
 * @see OutputOperator
 */
public class EfficientNetClassifier extends AbstractImageClassifier {

    private static final String DEFAULT_MODEL_ID = "inference4j/efficientnet-lite4";

    private EfficientNetClassifier(InferenceSession session,
                                   Preprocessor<BufferedImage, Tensor> imagePreprocessor,
                                   Labels labels, String inputName, int defaultTopK,
                                   OutputOperator outputOperator) {
        super(session, imagePreprocessor, labels, inputName, defaultTopK, outputOperator);
    }

    public static Builder builder() {
        return new Builder();
    }

    private static final float[] EFFICIENTNET_MEAN = {127f / 255f, 127f / 255f, 127f / 255f};
    private static final float[] EFFICIENTNET_STD = {128f / 255f, 128f / 255f, 128f / 255f};

    public static class Builder extends AbstractBuilder<Builder> {

        @Override
        protected Builder self() {
            return this;
        }

        public EfficientNetClassifier build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id);
                loadFromDirectory(dir);
            }
            validate();
            return new EfficientNetClassifier(session, preprocessor, labels, inputName, defaultTopK, outputOperator);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            try {
                if (this.inputName == null) {
                    this.inputName = session.inputNames().iterator().next();
                }

                Path labelsPath = dir.resolve("labels.txt");
                if (Files.exists(labelsPath)) {
                    this.labels = Labels.fromFile(labelsPath);
                }

                this.preprocessor = detectPipeline(session, this.inputName,
                        EFFICIENTNET_MEAN, EFFICIENTNET_STD);
                this.outputOperator = OutputOperator.identity();
            } catch (Exception e) {
                this.session.close();
                this.session = null;
                throw e;
            }
        }
    }
}
