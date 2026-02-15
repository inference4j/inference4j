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

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class ClipTextEncoderModelTest {

    @Test
    void encode_returnsNonEmptyFiniteEmbedding() {
        try (var encoder = ClipTextEncoder.builder().build()) {
            float[] embedding = encoder.encode("a photo of a cat");

            assertTrue(embedding.length > 0, "Embedding should be non-empty");
            for (int i = 0; i < embedding.length; i++) {
                assertTrue(Float.isFinite(embedding[i]),
                        "Embedding value at index " + i + " should be finite, got: " + embedding[i]);
            }
        }
    }

    @Test
    void encode_returns512Dimensions() {
        try (var encoder = ClipTextEncoder.builder().build()) {
            float[] embedding = encoder.encode("a photo of a cat");

            assertEquals(512, embedding.length, "CLIP ViT-B/32 should produce 512-dim embeddings");
        }
    }

    @Test
    void encode_isL2Normalized() {
        try (var encoder = ClipTextEncoder.builder().build()) {
            float[] embedding = encoder.encode("a photo of a cat");

            float norm = 0f;
            for (float v : embedding) {
                norm += v * v;
            }
            norm = (float) Math.sqrt(norm);

            assertEquals(1.0f, norm, 1e-3f,
                    "Embedding should be L2-normalized (norm ≈ 1.0), got: " + norm);
        }
    }

    @Test
    void encode_similarTextsProduceCloserEmbeddings() {
        try (var encoder = ClipTextEncoder.builder().build()) {
            float[] embCat = encoder.encode("a photo of a cat");
            float[] embKitten = encoder.encode("a picture of a kitten");
            float[] embCar = encoder.encode("a photo of a sports car");

            float simCatKitten = dot(embCat, embKitten);
            float simCatCar = dot(embCat, embCar);

            assertTrue(simCatKitten > simCatCar,
                    "Cat-kitten should be more similar than cat-car: " +
                            "cat-kitten=" + simCatKitten + " cat-car=" + simCatCar);
        }
    }

    @Test
    void crossModal_catImageMatchesCatText() throws IOException {
        BufferedImage catImage = ImageIO.read(
                ClipTextEncoderModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));

        try (var imageEncoder = ClipImageEncoder.builder().build();
             var textEncoder = ClipTextEncoder.builder().build()) {

            float[] imageEmb = imageEncoder.encode(catImage);
            float[] catText = textEncoder.encode("a photo of a cat");
            float[] dogText = textEncoder.encode("a photo of a dog");

            float catScore = dot(imageEmb, catText);
            float dogScore = dot(imageEmb, dogText);

            assertTrue(catScore > dogScore,
                    "Cat image should match 'cat' text better than 'dog' text: " +
                            "cat=" + catScore + " dog=" + dogScore);
        }
    }

    private static float dot(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
