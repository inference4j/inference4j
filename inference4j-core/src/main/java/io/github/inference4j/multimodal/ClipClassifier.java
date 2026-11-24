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
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.vision.Classification;
import io.github.inference4j.model.AbstractModelBuilder;
import io.github.inference4j.model.ModelFiles;

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
 * <h2>Confidences</h2>
 * <p>Like CLIP itself, the classifier multiplies the cosine similarities by the model's learned logit scale
 * before the softmax ({@code logit_scale.exp() * image_embeds @ text_embeds.T} in the reference
 * implementation). The default, {@value Builder#DEFAULT_LOGIT_SCALE}, is the trained value of the OpenAI CLIP
 * checkpoints; set {@link Builder#logitScale(float)} for a checkpoint trained to a different value.
 *
 * @see ClipImageEncoder
 * @see ClipTextEncoder
 * @see ZeroShotClassifier
 */
public class ClipClassifier implements ZeroShotClassifier<BufferedImage, Classification> {

    private final ClipImageEncoder imageEncoder;
    private final ClipTextEncoder textEncoder;
    private final float logitScale;

    private ClipClassifier(ClipImageEncoder imageEncoder, ClipTextEncoder textEncoder, float logitScale) {
        this.imageEncoder = imageEncoder;
        this.textEncoder = textEncoder;
        this.logitScale = logitScale;
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
        return toClassifications(imageEmbedding, labelEmbeddings, candidateLabels, topK, logitScale);
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
        ModelFiles.closeAll(imageEncoder, textEncoder);
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
                                                   List<String> labels, int topK, float logitScale) {
        // Both embeddings are L2-normalized, so the dot product is the cosine similarity
        float[] logits = new float[labelEmbeddings.length];
        for (int i = 0; i < labelEmbeddings.length; i++) {
            logits[i] = logitScale * MathOps.dotProduct(imageEmbedding, labelEmbeddings[i]);
        }

        float[] probabilities = MathOps.softmax(logits);
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

    public static class Builder extends AbstractModelBuilder<ClipClassifier, Builder> {
        private static final String DEFAULT_MODEL_ID = "inference4j/clip-vit-base-patch32";
        private static final List<String> MODEL_FILES =
                List.of("vision_model.onnx", "text_model.onnx", "vocab.json", "merges.txt");

        /**
         * The trained {@code logit_scale.exp()} of the OpenAI CLIP checkpoints (training clamps it at 100).
         */
        public static final float DEFAULT_LOGIT_SCALE = 100f;

        private float logitScale = DEFAULT_LOGIT_SCALE;

        public Builder() {
            super(DEFAULT_MODEL_ID);
        }

        /**
         * Sets the factor applied to the cosine similarities before the softmax: the model's
         * {@code logit_scale.exp()}. Defaults to {@value #DEFAULT_LOGIT_SCALE}, the trained value of the
         * OpenAI CLIP checkpoints. Only the confidences change; the ranking does not.
         *
         * @param logitScale a positive, finite scale
         * @return this builder
         */
        public Builder logitScale(float logitScale) {
            if (!(logitScale > 0f) || Float.isInfinite(logitScale)) {
                throw new IllegalArgumentException("logitScale must be positive and finite, got: " + logitScale);
            }
            this.logitScale = logitScale;
            return self();
        }

        @Override
        public ClipClassifier build() {
            // Both encoders read one opened snapshot, so an update between them cannot mix two revisions
            return load(MODEL_FILES, files -> new ClipClassifier(
                    ClipImageEncoder.create(files, null), ClipTextEncoder.create(files, null, null), logitScale));
        }
    }
}
