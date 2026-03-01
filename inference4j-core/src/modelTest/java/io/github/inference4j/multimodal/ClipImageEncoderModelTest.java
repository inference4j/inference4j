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
class ClipImageEncoderModelTest {

    private ClipImageEncoder encoder;
    private BufferedImage catImage;

    @BeforeAll
    void setUp() throws IOException {
        encoder = ClipImageEncoder.builder().build();
        catImage = ImageIO.read(ClipImageEncoderModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @AfterAll
    void tearDown() throws Exception {
        if (encoder != null) encoder.close();
    }

    @Test
    void encode_catImage_returnsNonEmptyFiniteEmbedding() {
        float[] embedding = encoder.encode(catImage);

        assertThat(embedding.length > 0).as("Embedding should be non-empty").isTrue();
        for (int i = 0; i < embedding.length; i++) {
            assertThat(Float.isFinite(embedding[i])).as("Embedding value at index " + i + " should be finite, got: " + embedding[i]).isTrue();
        }
    }

    @Test
    void encode_catImage_returns512Dimensions() {
        float[] embedding = encoder.encode(catImage);

        assertThat(embedding.length).as("CLIP ViT-B/32 should produce 512-dim embeddings").isEqualTo(512);
    }

    @Test
    void encode_catImage_isL2Normalized() {
        float[] embedding = encoder.encode(catImage);

        float norm = 0f;
        for (float v : embedding) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);

        assertThat(norm).as("Embedding should be L2-normalized (norm ≈ 1.0), got: " + norm).isCloseTo(1.0f, within(1e-3f));
    }
}
