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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.MathOps;
import io.github.inference4j.ModelSource;
import io.github.inference4j.OutputOperator;
import io.github.inference4j.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.image.ImageLayout;
import io.github.inference4j.image.ImageTransformPipeline;
import io.github.inference4j.image.Labels;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Base class for image classification models (ResNet, EfficientNet, etc.).
 *
 * <p>Provides shared inference logic: image loading, preprocessing via
 * {@link ImageTransformPipeline}, ONNX inference, and softmax post-processing.
 * Subclasses supply a {@code builder()} factory method with model-specific defaults.
 */
public abstract class AbstractImageClassifier implements ImageClassifier {

    protected final InferenceSession session;
    protected final ImageTransformPipeline pipeline;
    protected final Labels labels;
    protected final String inputName;
    protected final int defaultTopK;
    protected final OutputOperator outputOperator;

    protected AbstractImageClassifier(InferenceSession session, ImageTransformPipeline pipeline,
                                      Labels labels, String inputName, int defaultTopK,
                                      OutputOperator outputOperator) {
        this.session = session;
        this.pipeline = pipeline;
        this.labels = labels;
        this.inputName = inputName;
        this.defaultTopK = defaultTopK;
        this.outputOperator = outputOperator;
    }

    @Override
    public List<Classification> classify(BufferedImage image) {
        return classify(image, defaultTopK);
    }

    @Override
    public List<Classification> classify(BufferedImage image, int topK) {
        Tensor inputTensor = pipeline.transform(image);

        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put(inputName, inputTensor);

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

    @Override
    public void close() {
        session.close();
    }

    static List<Classification> postProcess(float[] logits, Labels labels, int topK,
                                              OutputOperator outputOperator) {
        float[] probabilities = outputOperator.apply(logits);
        int[] topIndices = MathOps.topK(probabilities, topK);

        List<Classification> results = new ArrayList<>(topIndices.length);
        for (int idx : topIndices) {
            results.add(new Classification(labels.get(idx), idx, probabilities[idx]));
        }
        return results;
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
     * Base builder for image classifiers.
     *
     * @param <B> the concrete builder type (for self-returning setters)
     */
    protected abstract static class AbstractBuilder<B extends AbstractBuilder<B>> {
        protected InferenceSession session;
        protected ModelSource modelSource;
        protected String modelId;
        protected SessionConfigurer sessionConfigurer;
        protected ImageTransformPipeline pipeline = ImageTransformPipeline.imagenet(224);
        protected Labels labels = Labels.imagenet();
        protected String inputName;
        protected int defaultTopK = 5;
        protected OutputOperator outputOperator = OutputOperator.softmax();

        @SuppressWarnings("unchecked")
        protected B self() {
            return (B) this;
        }

        B session(InferenceSession session) {
            this.session = session;
            return self();
        }

        public B sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return self();
        }

        public B modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return self();
        }

        public B modelId(String modelId) {
            this.modelId = modelId;
            return self();
        }

        public B pipeline(ImageTransformPipeline pipeline) {
            this.pipeline = pipeline;
            return self();
        }

        public B labels(Labels labels) {
            this.labels = labels;
            return self();
        }

        public B inputName(String inputName) {
            this.inputName = inputName;
            return self();
        }

        public B defaultTopK(int defaultTopK) {
            this.defaultTopK = defaultTopK;
            return self();
        }

        public B outputOperator(OutputOperator outputOperator) {
            this.outputOperator = outputOperator;
            return self();
        }

        protected void validate() {
            if (session == null) {
                throw new IllegalStateException("InferenceSession is required");
            }
            if (inputName == null) {
                inputName = session.inputNames().iterator().next();
            }
        }
    }
}
