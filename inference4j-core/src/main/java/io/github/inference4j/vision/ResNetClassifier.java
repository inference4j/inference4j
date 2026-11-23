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

import io.github.inference4j.processing.OutputOperator;

import java.util.List;

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
 *         .modelId("my-org/my-resnet")
 *         .modelSource(new LocalModelSource(modelsDir))   // resolves modelsDir/<modelId>
 *         .sessionOptions(opts -> opts.addCUDA(0))
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

    private ResNetClassifier(Parts parts) {
        super(parts);
    }

    public static Builder builder() {
        return new Builder();
    }

    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    public static class Builder extends AbstractBuilder<ResNetClassifier, Builder> {

        public Builder() {
            super(DEFAULT_MODEL_ID);
        }

        @Override
        public ResNetClassifier build() {
            return load(List.of("model.onnx"), files ->
                    new ResNetClassifier(parts(files, IMAGENET_MEAN, IMAGENET_STD, OutputOperator.softmax())));
        }
    }
}
