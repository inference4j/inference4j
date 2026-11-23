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

import io.github.inference4j.AbstractInferenceTask;
import io.github.inference4j.PreprocessResult;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.model.AbstractModelBuilder;
import io.github.inference4j.model.ModelFiles;
import io.github.inference4j.processing.OutputOperator;
import io.github.inference4j.processing.Postprocessor;
import io.github.inference4j.processing.Preprocessor;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.preprocessing.image.ImageLayout;
import io.github.inference4j.preprocessing.image.ImageTransformPipeline;
import io.github.inference4j.preprocessing.image.Labels;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Base class for image classification models (ResNet, EfficientNet, etc.).
 *
 * <p>Extends {@link AbstractInferenceTask} to enforce the preprocess → infer → postprocess
 * pipeline. The {@link #run(BufferedImage)} method is {@code final} and uses the default
 * {@code topK}. The parameterized {@link #classify(BufferedImage, int)} overload composes
 * the same building blocks with a custom {@code topK}.
 *
 * <p>Subclasses supply a {@code builder()} factory method with model-specific defaults.
 */
public abstract class AbstractImageClassifier
        extends AbstractInferenceTask<BufferedImage, List<Classification>>
        implements ImageClassifier {

    protected final Preprocessor<BufferedImage, Tensor> imagePreprocessor;
    protected final Labels labels;
    protected final String inputName;
    protected final int defaultTopK;
    protected final OutputOperator outputOperator;

    protected AbstractImageClassifier(InferenceSession session,
                                      Preprocessor<BufferedImage, Tensor> imagePreprocessor,
                                      Labels labels, String inputName, int defaultTopK,
                                      OutputOperator outputOperator) {
        super(session,
                image -> PreprocessResult.of(Map.of(inputName, imagePreprocessor.process(image))),
                ctx -> {
                    Tensor outputTensor = ctx.outputs().values().iterator().next();
                    float[] logits = outputTensor.toFloats();
                    return postProcess(logits, labels, defaultTopK, outputOperator);
                });
        this.imagePreprocessor = imagePreprocessor;
        this.labels = labels;
        this.inputName = inputName;
        this.defaultTopK = defaultTopK;
        this.outputOperator = outputOperator;
    }

    @Override
    public List<Classification> classify(BufferedImage image) {
        return run(image);
    }

    @Override
    public List<Classification> classify(BufferedImage image, int topK) {
        Tensor inputTensor = imagePreprocessor.process(image);
        Map<String, Tensor> inputs = Map.of(inputName, inputTensor);
        Map<String, Tensor> outputs = session.run(inputs);
        Tensor outputTensor = outputs.values().iterator().next();
        float[] logits = outputTensor.toFloats();
        return postProcess(logits, labels, topK, outputOperator);
    }

    @Override
    public List<Classification> classify(Path imagePath) {
        return classify(imagePath, defaultTopK);
    }

    @Override
    public List<Classification> classify(Path imagePath, int topK) {
        BufferedImage image = loadImage(imagePath);
        return classify(image, topK);
    }

    static Postprocessor<float[], List<Classification>> classificationPostprocessor(
            Labels labels, int topK, OutputOperator outputOperator) {
        return logits -> {
            float[] probabilities = outputOperator.apply(logits);
            int[] topIndices = MathOps.topK(probabilities, topK);

            List<Classification> results = new ArrayList<>(topIndices.length);
            for (int idx : topIndices) {
                results.add(new Classification(labels.get(idx), idx, probabilities[idx]));
            }
            return results;
        };
    }

    static List<Classification> postProcess(float[] logits, Labels labels, int topK,
                                              OutputOperator outputOperator) {
        return classificationPostprocessor(labels, topK, outputOperator).process(logits);
    }

    protected static BufferedImage loadImage(Path path) {
        try {
            BufferedImage image = ImageIO.read(path.toFile());
            if (image == null) {
                throw new InferenceException("Unsupported image format: " + path);
            }
            return image;
        } catch (IOException e) {
            throw new InferenceException("Failed to read image: " + path, e);
        }
    }

    protected static ImageTransformPipeline detectPipeline(InferenceSession session,
                                                              String inputName,
                                                              float[] mean, float[] std) {
        long[] shape = session.inputShape(inputName);
        ImageLayout layout = ImageLayout.detect(shape);
        int size = layout.imageSize(shape);
        return ImageTransformPipeline.builder()
                .resize(size, size)
                .centerCrop(size, size)
                .mean(mean)
                .std(std)
                .layout(layout)
                .build();
    }

    /**
     * The parts an image classifier is built from.
     *
     * @param session           the session, which the classifier owns
     * @param imagePreprocessor turns an image into the model input
     * @param labels            class names, by class index
     * @param inputName         the model input the image goes to
     * @param defaultTopK       how many classes {@link #classify(BufferedImage)} returns
     * @param outputOperator    turns the model output into probabilities
     */
    protected record Parts(InferenceSession session, Preprocessor<BufferedImage, Tensor> imagePreprocessor,
                           Labels labels, String inputName, int defaultTopK, OutputOperator outputOperator) {
    }

    /**
     * Creates a classifier from its parts.
     *
     * @param parts the parts, usually from {@link AbstractBuilder#parts}
     */
    protected AbstractImageClassifier(Parts parts) {
        this(parts.session(), parts.imagePreprocessor(), parts.labels(), parts.inputName(), parts.defaultTopK(),
                parts.outputOperator());
    }

    /**
     * Base builder for image classifiers.
     *
     * @param <T> the classifier type
     * @param <B> the concrete builder type (for self-returning setters)
     */
    protected abstract static class AbstractBuilder<T extends AbstractImageClassifier, B extends AbstractBuilder<T, B>>
            extends AbstractModelBuilder<T, B> {
        protected Preprocessor<BufferedImage, Tensor> preprocessor;
        protected Labels labels;
        protected String inputName;
        protected int defaultTopK = 5;
        protected OutputOperator outputOperator;

        /**
         * @param defaultModelId the model used when {@link #modelId(String)} is not set
         */
        protected AbstractBuilder(String defaultModelId) {
            super(defaultModelId);
        }

        /**
         * Turns an image into the model input. Defaults to resizing to the model's input size and normalizing with
         * the wrapper's mean and standard deviation, in the model's layout.
         */
        public B preprocessor(Preprocessor<BufferedImage, Tensor> preprocessor) {
            this.preprocessor = preprocessor;
            return self();
        }

        /** Class names, by class index. Defaults to a {@code labels.txt} next to the model, or ImageNet. */
        public B labels(Labels labels) {
            this.labels = labels;
            return self();
        }

        /** The model input the image goes to. Defaults to the model's first input. */
        public B inputName(String inputName) {
            this.inputName = inputName;
            return self();
        }

        /** How many classes {@code classify(image)} returns. Defaults to 5. */
        public B defaultTopK(int defaultTopK) {
            this.defaultTopK = defaultTopK;
            return self();
        }

        /** Turns the model output into probabilities. The default depends on the wrapper. */
        public B outputOperator(OutputOperator outputOperator) {
            this.outputOperator = outputOperator;
            return self();
        }

        /**
         * The classifier's parts: {@code model.onnx} and the options set on this builder, with defaults for those
         * not set.
         *
         * @param files                 the model's files
         * @param mean                  per-channel mean of the default preprocessor
         * @param std                   per-channel standard deviation of the default preprocessor
         * @param defaultOutputOperator the output operator when none is set
         * @return the parts
         */
        protected Parts parts(ModelFiles files, float[] mean, float[] std, OutputOperator defaultOutputOperator) {
            InferenceSession session = files.session("model.onnx");
            String input = inputName != null ? inputName : session.inputNames().iterator().next();
            Preprocessor<BufferedImage, Tensor> imagePreprocessor = preprocessor != null
                    ? preprocessor : detectPipeline(session, input, mean, std);
            Labels resolvedLabels = labels != null
                    ? labels : files.optional("labels.txt").map(Labels::fromFile).orElseGet(Labels::imagenet);
            OutputOperator operator = outputOperator != null ? outputOperator : defaultOutputOperator;
            return new Parts(session, imagePreprocessor, resolvedLabels, input, defaultTopK, operator);
        }
    }
}
