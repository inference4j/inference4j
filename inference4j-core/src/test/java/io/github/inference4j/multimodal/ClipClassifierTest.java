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

import io.github.inference4j.TestModels;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.vision.Classification;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

class ClipClassifierTest {

    private static final float SCALE = ClipClassifier.Builder.DEFAULT_LOGIT_SCALE;

    @Test
    void toClassifications_returnsTopKSortedByConfidence() {
        // Image embedding pointing toward label 0
        float[] imageEmb = {1.0f, 0.0f};

        // Label embeddings: label 0 is aligned, label 1 is orthogonal, label 2 is opposite
        float[][] labelEmbs = {
                {1.0f, 0.0f},  // dot = 1.0
                {0.0f, 1.0f},  // dot = 0.0
                {-1.0f, 0.0f}, // dot = -1.0
        };
        List<String> labels = List.of("cat", "dog", "car");

        List<Classification> results = ClipClassifier.toClassifications(imageEmb, labelEmbs, labels, 3, SCALE);

        assertThat(results).hasSize(3);
        assertThat(results.get(0).label()).isEqualTo("cat");
        assertThat(results.get(0).index()).isEqualTo(0);
        assertThat(results.get(1).label()).isEqualTo("dog");
        assertThat(results.get(1).index()).isEqualTo(1);
        assertThat(results.get(2).label()).isEqualTo("car");
        assertThat(results.get(2).index()).isEqualTo(2);
    }

    @Test
    void toClassifications_topKLimitsResults() {
        float[] imageEmb = {1.0f, 0.0f};
        float[][] labelEmbs = {
                {1.0f, 0.0f},
                {0.0f, 1.0f},
                {-1.0f, 0.0f},
        };
        List<String> labels = List.of("cat", "dog", "car");

        List<Classification> results = ClipClassifier.toClassifications(imageEmb, labelEmbs, labels, 1, SCALE);

        assertThat(results).hasSize(1);
        assertThat(results.get(0).label()).isEqualTo("cat");
    }

    @Test
    void toClassifications_confidencesSumToOne() {
        float[] imageEmb = MathOps.l2Normalize(new float[]{0.7f, 0.3f});
        float[][] labelEmbs = {
                MathOps.l2Normalize(new float[]{1.0f, 0.0f}),
                MathOps.l2Normalize(new float[]{0.0f, 1.0f}),
                MathOps.l2Normalize(new float[]{0.5f, 0.5f}),
        };
        List<String> labels = List.of("cat", "dog", "bird");

        List<Classification> results = ClipClassifier.toClassifications(imageEmb, labelEmbs, labels, 3, SCALE);

        float sum = 0f;
        for (Classification c : results) {
            assertThat(c.confidence()).isGreaterThan(0f);
            assertThat(c.confidence()).isLessThanOrEqualTo(1f);
            sum += c.confidence();
        }
        assertThat(sum).isCloseTo(1.0f, within(1e-5f));
    }

    @Test
    void toClassifications_sortedDescending() {
        float[] imageEmb = MathOps.l2Normalize(new float[]{0.5f, 0.5f});
        float[][] labelEmbs = {
                MathOps.l2Normalize(new float[]{1.0f, 0.0f}),
                MathOps.l2Normalize(new float[]{0.0f, 1.0f}),
                MathOps.l2Normalize(new float[]{0.5f, 0.5f}),
                MathOps.l2Normalize(new float[]{-1.0f, 0.0f}),
        };
        List<String> labels = List.of("a", "b", "c", "d");

        List<Classification> results = ClipClassifier.toClassifications(imageEmb, labelEmbs, labels, 4, SCALE);

        for (int i = 1; i < results.size(); i++) {
            assertThat(results.get(i - 1).confidence())
                    .as("Results not sorted descending at index " + i)
                    .isGreaterThanOrEqualTo(results.get(i).confidence());
        }
    }

    // End to end through the model files: the fixture vision model averages each normalized channel, the
    // fixture text model counts the tokens r</w>, g</w> and b</w>, so a red image is closest to "r".

    @TempDir
    Path dir;

    @Test
    void classify_usesEncodersAndReturnsClassifications() {
        try (ClipClassifier classifier = classifier()) {
            List<Classification> red = classifier.classify(filled(0xFF0000), List.of("r", "g", "b"));
            List<Classification> green = classifier.classify(filled(0x00FF00), List.of("r", "g", "b"));

            assertThat(red).hasSize(3);
            assertThat(red.get(0).label()).isEqualTo("r");
            assertThat(red.get(0).index()).isZero();
            assertThat(green.get(0).label()).isEqualTo("g");
        }
    }

    @Test
    void defaultLogitScaleGivesAConfidentTopLabel() {
        try (ClipClassifier classifier = classifier()) {
            List<Classification> red = classifier.classify(filled(0xFF0000), List.of("r", "g", "b"));

            // Unscaled, the cosine similarities would spread the mass over all three labels
            assertThat(red.get(0).confidence()).isGreaterThan(0.99f);
            assertThat(red.stream().map(Classification::confidence).reduce(0f, Float::sum))
                    .isCloseTo(1.0f, within(1e-5f));
        }
    }

    @Test
    void lowerLogitScaleFlattensConfidencesButKeepsTopLabel() {
        TestModels.install(dir, "clip-vision", "vision_model.onnx");
        TestModels.install(dir, "clip-text", "text_model.onnx");
        TestModels.clipVocabulary(dir);
        try (ClipClassifier scaled = ClipClassifier.builder().modelSource(TestModels.source(dir)).build();
             ClipClassifier unscaled = ClipClassifier.builder().modelSource(TestModels.source(dir))
                     .logitScale(1f).build()) {
            List<String> labels = List.of("r", "g", "b");
            List<Classification> sharp = scaled.classify(filled(0xFF0000), labels);
            List<Classification> flat = unscaled.classify(filled(0xFF0000), labels);

            assertThat(flat.get(0).label()).isEqualTo(sharp.get(0).label()).isEqualTo("r");
            assertThat(flat.get(0).confidence()).isLessThan(0.9f).isLessThan(sharp.get(0).confidence());
        }
    }

    @Test
    void logitScaleMustBePositiveAndFinite() {
        for (float invalid : new float[]{0f, -1f, Float.NaN, Float.POSITIVE_INFINITY}) {
            assertThatThrownBy(() -> ClipClassifier.builder().logitScale(invalid))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("logitScale");
        }
    }

    @Test
    void classify_withEmptyCandidateLabels_returnsEmpty() {
        try (ClipClassifier classifier = classifier()) {
            assertThat(classifier.classify(filled(0xFF0000), List.of())).isEmpty();
        }
    }

    @Test
    void closedClassifierRejectsClassify() {
        ClipClassifier classifier = classifier();

        classifier.close();

        assertThatThrownBy(() -> classifier.classify(filled(0xFF0000), List.of("r", "g")))
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void everyModelFileIsRequired() {
        TestModels.install(dir, "clip-vision", "vision_model.onnx");
        TestModels.install(dir, "clip-text", "text_model.onnx");

        assertThatThrownBy(() -> ClipClassifier.builder().modelSource(TestModels.source(dir)).build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("vocab.json");
    }

    private ClipClassifier classifier() {
        TestModels.install(dir, "clip-vision", "vision_model.onnx");
        TestModels.install(dir, "clip-text", "text_model.onnx");
        TestModels.clipVocabulary(dir);
        return ClipClassifier.builder().modelSource(TestModels.source(dir)).build();
    }

    private static BufferedImage filled(int rgb) {
        BufferedImage image = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }
}
