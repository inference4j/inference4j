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

import io.github.inference4j.ZeroShotClassifier;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.vision.Classification;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Zero-shot image classifier powered by CLIP — classify images using arbitrary
 * text labels with no training required.
 *
 * <p>Wraps {@link ClipImageEncoder} and {@link ClipTextEncoder} to provide a
 * familiar {@link ZeroShotClassifier} API. Labels are provided per call,
 * allowing the same classifier instance to be reused for different label sets
 * without rebuilding ONNX sessions.
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (ClipClassifier classifier = ClipClassifier.builder().build()) {
 *     List<Classification> results = classifier.classify(
 *             Path.of("photo.jpg"), List.of("cat", "dog", "bird"));
 *     System.out.println(results.get(0).label()); // "cat"
 * }
 * }</pre>
 *
 * <h2>Prompt tips</h2>
 * <p>CLIP was trained on natural language captions, so passing full prompt text
 * as labels produces better results. For example, instead of {@code "cat"},
 * pass {@code "a photo of a cat"}.
 *
 * @see ClipImageEncoder
 * @see ClipTextEncoder
 * @see ZeroShotClassifier
 */
public class ClipClassifier implements ZeroShotClassifier<BufferedImage, Classification> {

    private final ClipImageEncoder imageEncoder;
    private final ClipTextEncoder textEncoder;

    private ClipClassifier(ClipImageEncoder imageEncoder, ClipTextEncoder textEncoder) {
        this.imageEncoder = imageEncoder;
        this.textEncoder = textEncoder;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<Classification> classify(BufferedImage image, List<String> candidateLabels) {
        return classify(image, candidateLabels, candidateLabels.size());
    }

    public List<Classification> classify(BufferedImage image, List<String> candidateLabels, int topK) {
        if (candidateLabels.isEmpty()) {
            return List.of();
        }
        float[] imageEmbedding = imageEncoder.encode(image);
        float[][] labelEmbeddings = encodeLabels(candidateLabels);
        return toClassifications(imageEmbedding, labelEmbeddings, candidateLabels, topK);
    }

    public List<Classification> classify(Path imagePath, List<String> candidateLabels) {
        return classify(imagePath, candidateLabels, candidateLabels.size());
    }

    public List<Classification> classify(Path imagePath, List<String> candidateLabels, int topK) {
        BufferedImage image = loadImage(imagePath);
        return classify(image, candidateLabels, topK);
    }

    @Override
    public void close() {
        imageEncoder.close();
        textEncoder.close();
    }

    private float[][] encodeLabels(List<String> labels) {
        float[][] embeddings = new float[labels.size()][];
        for (int i = 0; i < labels.size(); i++) {
            embeddings[i] = textEncoder.encode(labels.get(i));
        }
        return embeddings;
    }

    static List<Classification> toClassifications(float[] imageEmbedding,
                                                   float[][] labelEmbeddings,
                                                   List<String> labels, int topK) {
        float[] similarities = new float[labelEmbeddings.length];
        for (int i = 0; i < labelEmbeddings.length; i++) {
            similarities[i] = MathOps.dotProduct(imageEmbedding, labelEmbeddings[i]);
        }

        float[] probabilities = MathOps.softmax(similarities);
        int[] topIndices = MathOps.topK(probabilities, topK);

        List<Classification> results = new ArrayList<>(topIndices.length);
        for (int idx : topIndices) {
            results.add(new Classification(labels.get(idx), idx, probabilities[idx]));
        }
        return results;
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

        public ClipClassifier build() {
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

            return new ClipClassifier(imageEncoder, textEncoder);
        }
    }
}
