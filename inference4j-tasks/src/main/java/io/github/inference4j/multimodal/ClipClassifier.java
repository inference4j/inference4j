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

package io.github.inference4j.multimodal;

import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.vision.Classification;
import io.github.inference4j.vision.ImageClassifier;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Zero-shot image classifier powered by CLIP — classify images using arbitrary
 * text labels with no training required.
 *
 * <p>Wraps {@link ClipImageEncoder} and {@link ClipTextEncoder} to provide a
 * familiar {@link ImageClassifier} API. At build time, label text is encoded
 * into embeddings using the prompt template (e.g. "a photo of a cat"). At
 * classify time, the image is encoded and compared against all label embeddings
 * via dot-product similarity.
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (ClipClassifier classifier = ClipClassifier.builder()
 *         .labels("cat", "dog", "bird", "car", "airplane")
 *         .build()) {
 *     List<Classification> results = classifier.classify(Path.of("photo.jpg"));
 *     System.out.println(results.get(0).label()); // "cat"
 * }
 * }</pre>
 *
 * <h2>Custom prompt template</h2>
 * <pre>{@code
 * ClipClassifier classifier = ClipClassifier.builder()
 *         .labels("happy", "sad", "angry", "surprised")
 *         .promptTemplate("a photo of a {} person")
 *         .build();
 * }</pre>
 *
 * @see ClipImageEncoder
 * @see ClipTextEncoder
 * @see ImageClassifier
 */
public class ClipClassifier implements ImageClassifier {

    private static final String DEFAULT_PROMPT_TEMPLATE = "a photo of a {}";

    private final ClipImageEncoder imageEncoder;
    private final ClipTextEncoder textEncoder;
    private final float[][] labelEmbeddings;
    private final List<String> labels;
    private final int defaultTopK;

    private ClipClassifier(ClipImageEncoder imageEncoder, ClipTextEncoder textEncoder,
                           float[][] labelEmbeddings, List<String> labels, int defaultTopK) {
        this.imageEncoder = imageEncoder;
        this.textEncoder = textEncoder;
        this.labelEmbeddings = labelEmbeddings;
        this.labels = labels;
        this.defaultTopK = defaultTopK;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<Classification> classify(BufferedImage image) {
        return classify(image, defaultTopK);
    }

    @Override
    public List<Classification> classify(BufferedImage image, int topK) {
        float[] imageEmbedding = imageEncoder.encode(image);
        return toClassifications(imageEmbedding, labelEmbeddings, labels, topK);
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
        imageEncoder.close();
        textEncoder.close();
    }

    static List<Classification> toClassifications(float[] imageEmbedding,
                                                   float[][] labelEmbeddings,
                                                   List<String> labels, int topK) {
        float[] similarities = new float[labelEmbeddings.length];
        for (int i = 0; i < labelEmbeddings.length; i++) {
            similarities[i] = dot(imageEmbedding, labelEmbeddings[i]);
        }

        float[] probabilities = MathOps.softmax(similarities);
        int[] topIndices = MathOps.topK(probabilities, topK);

        List<Classification> results = new ArrayList<>(topIndices.length);
        for (int idx : topIndices) {
            results.add(new Classification(labels.get(idx), idx, probabilities[idx]));
        }
        return results;
    }

    private static float dot(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static BufferedImage loadImage(Path path) {
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

    public static class Builder {
        private ClipImageEncoder imageEncoder;
        private ClipTextEncoder textEncoder;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private List<String> labels;
        private String promptTemplate = DEFAULT_PROMPT_TEMPLATE;
        private int defaultTopK = -1;

        Builder imageEncoder(ClipImageEncoder imageEncoder) {
            this.imageEncoder = imageEncoder;
            return this;
        }

        Builder textEncoder(ClipTextEncoder textEncoder) {
            this.textEncoder = textEncoder;
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

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public Builder labels(String... labels) {
            this.labels = Arrays.asList(labels);
            return this;
        }

        public Builder labels(List<String> labels) {
            this.labels = List.copyOf(labels);
            return this;
        }

        public Builder promptTemplate(String promptTemplate) {
            this.promptTemplate = promptTemplate;
            return this;
        }

        public Builder defaultTopK(int defaultTopK) {
            this.defaultTopK = defaultTopK;
            return this;
        }

        public ClipClassifier build() {
            if (labels == null || labels.isEmpty()) {
                throw new IllegalStateException("Labels are required for zero-shot classification");
            }

            if (imageEncoder == null) {
                ClipImageEncoder.Builder imageBuilder = ClipImageEncoder.builder();
                if (modelSource != null) imageBuilder.modelSource(modelSource);
                if (modelId != null) imageBuilder.modelId(modelId);
                if (sessionConfigurer != null) imageBuilder.sessionOptions(sessionConfigurer);
                imageEncoder = imageBuilder.build();
            }

            try {
                if (textEncoder == null) {
                    ClipTextEncoder.Builder textBuilder = ClipTextEncoder.builder();
                    if (modelSource != null) textBuilder.modelSource(modelSource);
                    if (modelId != null) textBuilder.modelId(modelId);
                    if (sessionConfigurer != null) textBuilder.sessionOptions(sessionConfigurer);
                    textEncoder = textBuilder.build();
                }
            } catch (Exception e) {
                imageEncoder.close();
                throw e;
            }

            int topK = defaultTopK > 0 ? defaultTopK : labels.size();

            try {
                float[][] embeddings = new float[labels.size()][];
                for (int i = 0; i < labels.size(); i++) {
                    String prompt = promptTemplate.replace("{}", labels.get(i));
                    embeddings[i] = textEncoder.encode(prompt);
                }
                return new ClipClassifier(imageEncoder, textEncoder, embeddings, labels, topK);
            } catch (Exception e) {
                imageEncoder.close();
                textEncoder.close();
                throw e;
            }
        }
    }
}
