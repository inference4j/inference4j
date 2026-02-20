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

import io.github.inference4j.vision.Classification;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ClipClassifierModelTest {

    private static BufferedImage loadCatImage() throws IOException {
        return ImageIO.read(ClipClassifierModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @Test
    void classify_catImage_catIsTopLabel() throws IOException {
        try (io.github.inference4j.multimodal.ClipClassifier classifier = ClipClassifier.builder().build()) {
            List<Classification> results = classifier.classify(
                    loadCatImage(),
                    List.of("a photo of a cat", "a photo of a dog", "a photo of a bird",
                            "a photo of a car", "a photo of an airplane"));

            assertFalse(results.isEmpty());
            assertEquals("a photo of a cat", results.get(0).label(),
                    "Expected 'a photo of a cat' as top label, got: " + results.get(0).label());
            assertTrue(results.get(0).confidence() > 0.15f,
                    "Expected cat confidence > 0.15, got: " + results.get(0).confidence());
        }
    }

    @Test
    void classify_catImage_confidencesSumToOne() throws IOException {
        try (ClipClassifier classifier = ClipClassifier.builder().build()) {
            List<Classification> results = classifier.classify(
                    loadCatImage(),
                    List.of("a photo of a cat", "a photo of a dog", "a photo of a bird"));

            float sum = 0f;
            for (Classification c : results) {
                assertTrue(c.confidence() > 0f);
                sum += c.confidence();
            }
            assertEquals(1.0f, sum, 1e-3f);
        }
    }

    @Test
    void classify_catImage_withTopK() throws IOException {
        try (ClipClassifier classifier = ClipClassifier.builder().build()) {
            List<Classification> results = classifier.classify(
                    loadCatImage(),
                    List.of("a photo of a cat", "a photo of a dog", "a photo of a bird",
                            "a photo of a car", "a photo of an airplane"),
                    2);

            assertEquals(2, results.size());
        }
    }
}
