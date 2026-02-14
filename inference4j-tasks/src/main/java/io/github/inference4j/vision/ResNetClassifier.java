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
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.image.ImageTransformPipeline;
import io.github.inference4j.image.Labels;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Image classifier for ResNet architectures.
 *
 * <h2>Tested model</h2>
 * <p>This wrapper is tested against
 * <a href="https://huggingface.co/onnxmodelzoo/resnet50-v1-7">ResNet-50 v1.7</a>
 * (ONNX Model Zoo). That model outputs <strong>raw logits</strong> (no activation on
 * the final layer), so this wrapper applies {@link OutputOperator#softmax()} by default
 * to produce probabilities.
 *
 * <p><strong>Other ResNet exports may differ.</strong> Some exporters (e.g., PyTorch's
 * {@code torch.onnx.export} with a model that includes {@code F.softmax}) bake softmax
 * into the graph. Using this wrapper with such a model would apply softmax twice,
 * silently compressing confidence scores. If your model already outputs probabilities,
 * override with {@code .outputOperator(OutputOperator.identity())} via the builder.
 *
 * <h2>Preprocessing</h2>
 * <ul>
 *   <li>Normalization: ImageNet mean {@code [0.485, 0.456, 0.406]}, std {@code [0.229, 0.224, 0.225]}</li>
 *   <li>Input layout: auto-detected from model (typically NCHW)</li>
 *   <li>Input size: auto-detected from model (typically 224x224)</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (ResNetClassifier classifier = ResNetClassifier.builder().build()) {
 *     List<Classification> results = classifier.classify(Path.of("cat.jpg"));
 * }
 * }</pre>
 *
 * <h2>Custom configuration</h2>
 * <pre>{@code
 * try (ResNetClassifier classifier = ResNetClassifier.builder()
 *         .session(InferenceSession.create(modelPath))
 *         .pipeline(ImageTransformPipeline.imagenet(224))
 *         .labels(Labels.fromFile(Path.of("my-labels.txt")))
 *         .outputOperator(OutputOperator.identity()) // if model has built-in softmax
 *         .defaultTopK(10)
 *         .build()) {
 *     List<Classification> results = classifier.classify(image, 3);
 * }
 * }</pre>
 *
 * @see OutputOperator
 */
public class ResNetClassifier extends AbstractImageClassifier {

    private static final String DEFAULT_MODEL_ID = "inference4j/resnet50-v1-7";

    private ResNetClassifier(InferenceSession session, ImageTransformPipeline pipeline,
                             Labels labels, String inputName, int defaultTopK,
                             OutputOperator outputOperator) {
        super(session, pipeline, labels, inputName, defaultTopK, outputOperator);
    }

    public static Builder builder() {
        return new Builder();
    }

    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    public static class Builder extends AbstractBuilder<Builder> {

        @Override
        protected Builder self() {
            return this;
        }

        public ResNetClassifier build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id);
                loadFromDirectory(dir);
            }
            validate();
            return new ResNetClassifier(session, pipeline, labels, inputName, defaultTopK, outputOperator);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }

            this.session = InferenceSession.create(modelPath);
            try {
                if (this.inputName == null) {
                    this.inputName = session.inputNames().iterator().next();
                }

                Path labelsPath = dir.resolve("labels.txt");
                if (Files.exists(labelsPath)) {
                    this.labels = Labels.fromFile(labelsPath);
                }

                this.pipeline = detectPipeline(session, this.inputName,
                        IMAGENET_MEAN, IMAGENET_STD);
                this.outputOperator = OutputOperator.softmax();
            } catch (Exception e) {
                this.session.close();
                this.session = null;
                throw e;
            }
        }
    }
}
