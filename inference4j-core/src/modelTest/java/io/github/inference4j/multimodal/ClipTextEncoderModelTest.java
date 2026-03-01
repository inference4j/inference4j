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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;

import static org.assertj.core.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ClipTextEncoderModelTest {

    private ClipTextEncoder textEncoder;
    private ClipImageEncoder imageEncoder;

    @BeforeAll
    void setUp() throws IOException {
        textEncoder = ClipTextEncoder.builder().build();
        imageEncoder = ClipImageEncoder.builder().build();
    }

    @AfterAll
    void tearDown() throws Exception {
        if (textEncoder != null) textEncoder.close();
        if (imageEncoder != null) imageEncoder.close();
    }

    @Test
    void encode_returnsNonEmptyFiniteEmbedding() {
        float[] embedding = textEncoder.encode("a photo of a cat");

        assertThat(embedding.length > 0).as("Embedding should be non-empty").isTrue();
        for (int i = 0; i < embedding.length; i++) {
            assertThat(Float.isFinite(embedding[i])).as("Embedding value at index " + i + " should be finite, got: " + embedding[i]).isTrue();
        }
    }

    @Test
    void encode_returns512Dimensions() {
        float[] embedding = textEncoder.encode("a photo of a cat");

        assertThat(embedding.length).as("CLIP ViT-B/32 should produce 512-dim embeddings").isEqualTo(512);
    }

    @Test
    void encode_isL2Normalized() {
        float[] embedding = textEncoder.encode("a photo of a cat");

        float norm = 0f;
        for (float v : embedding) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);

        assertThat(norm).as("Embedding should be L2-normalized (norm ≈ 1.0), got: " + norm).isCloseTo(1.0f, within(1e-3f));
    }

    @Test
    void encode_similarTextsProduceCloserEmbeddings() {
        float[] embCat = textEncoder.encode("a photo of a cat");
        float[] embKitten = textEncoder.encode("a picture of a kitten");
        float[] embCar = textEncoder.encode("a photo of a sports car");

        float simCatKitten = dot(embCat, embKitten);
        float simCatCar = dot(embCat, embCar);

        assertThat(simCatKitten > simCatCar).as("Cat-kitten should be more similar than cat-car: " +
                        "cat-kitten=" + simCatKitten + " cat-car=" + simCatCar).isTrue();
    }

    @Test
    void crossModal_catImageMatchesCatText() throws IOException {
        BufferedImage catImage = ImageIO.read(
                ClipTextEncoderModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));

        float[] imageEmb = imageEncoder.encode(catImage);
        float[] catText = textEncoder.encode("a photo of a cat");
        float[] dogText = textEncoder.encode("a photo of a dog");

        float catScore = dot(imageEmb, catText);
        float dogScore = dot(imageEmb, dogText);

        assertThat(catScore > dogScore).as("Cat image should match 'cat' text better than 'dog' text: " +
                        "cat=" + catScore + " dog=" + dogScore).isTrue();
    }

    private static float dot(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
