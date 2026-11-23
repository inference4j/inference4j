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

import io.github.inference4j.Tensor;
import io.github.inference4j.TestModels;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.Preprocessor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

class ClipImageEncoderTest {

    // The fixture model averages each channel, so a 1 x 3 x 1 x 1 input is its own raw embedding.

    /** Feeds the model the raw embedding [3, 4, 0], whatever the image. */
    private static final Preprocessor<BufferedImage, Tensor> THREE_FOUR_ZERO =
            image -> Tensor.fromFloats(new float[]{3f, 4f, 0f}, new long[]{1, 3, 1, 1});

    @TempDir
    Path dir;

    @Test
    void encode_returnsL2NormalizedEmbedding() {
        try (ClipImageEncoder encoder = encoder()) {
            float[] embedding = encoder.encode(new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB));

            // Raw output: [3, 4, 0] -> L2 norm = 5, normalized = [0.6, 0.8, 0]
            assertThat(embedding).containsExactly(new float[]{0.6f, 0.8f, 0f}, within(1e-5f));

            float norm = 0f;
            for (float v : embedding) {
                norm += v * v;
            }
            assertThat((float) Math.sqrt(norm)).isCloseTo(1.0f, within(1e-5f));
        }
    }

    @Test
    void encodeBatch_returnsOneEmbeddingPerImage() {
        try (ClipImageEncoder encoder = encoder()) {
            List<BufferedImage> images = List.of(
                    new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB),
                    new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB));
            List<float[]> results = encoder.encodeBatch(images);

            assertThat(results).hasSize(2);
            for (float[] emb : results) {
                assertThat(emb).hasSize(3);
            }
        }
    }

    @Test
    void closedEncoderRejectsEncode() {
        ClipImageEncoder encoder = encoder();

        encoder.close();

        assertThatThrownBy(() -> encoder.encode(new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB)))
                .isInstanceOf(IllegalStateException.class);
    }

    private ClipImageEncoder encoder() {
        TestModels.install(dir, "clip-vision", "vision_model.onnx");
        return ClipImageEncoder.builder()
                .modelSource(TestModels.source(dir))
                .preprocessor(THREE_FOUR_ZERO)
                .build();
    }

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                ClipImageEncoder.builder()
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void clipNormalization_usesCorrectMeanAndStd() {
        assertThat(ClipImageEncoder.CLIP_MEAN).containsExactly(
                new float[]{0.48145466f, 0.4578275f, 0.40821073f},
                within(1e-7f));
        assertThat(ClipImageEncoder.CLIP_STD).containsExactly(
                new float[]{0.26862954f, 0.26130258f, 0.27577711f},
                within(1e-7f));
    }
}
