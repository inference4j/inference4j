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

class ClipImageEncoderModelTest {

    private static BufferedImage loadCatImage() throws IOException {
        return ImageIO.read(ClipImageEncoderModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @Test
    void encode_catImage_returnsNonEmptyFiniteEmbedding() throws IOException {
        try (var encoder = ClipImageEncoder.builder().build()) {
            float[] embedding = encoder.encode(loadCatImage());

            assertTrue(embedding.length > 0, "Embedding should be non-empty");
            for (int i = 0; i < embedding.length; i++) {
                assertTrue(Float.isFinite(embedding[i]),
                        "Embedding value at index " + i + " should be finite, got: " + embedding[i]);
            }
        }
    }

    @Test
    void encode_catImage_returns512Dimensions() throws IOException {
        try (var encoder = ClipImageEncoder.builder().build()) {
            float[] embedding = encoder.encode(loadCatImage());

            assertEquals(512, embedding.length, "CLIP ViT-B/32 should produce 512-dim embeddings");
        }
    }

    @Test
    void encode_catImage_isL2Normalized() throws IOException {
        try (var encoder = ClipImageEncoder.builder().build()) {
            float[] embedding = encoder.encode(loadCatImage());

            float norm = 0f;
            for (float v : embedding) {
                norm += v * v;
            }
            norm = (float) Math.sqrt(norm);

            assertEquals(1.0f, norm, 1e-3f,
                    "Embedding should be L2-normalized (norm ≈ 1.0), got: " + norm);
        }
    }
}
