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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.Preprocessor;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ClipImageEncoderTest {

    @Test
    void encode_returnsL2NormalizedEmbedding() {
        InferenceSession session = mock(InferenceSession.class);
        @SuppressWarnings("unchecked")
        Preprocessor<BufferedImage, Tensor> preprocessor = mock(Preprocessor.class);

        Tensor inputTensor = Tensor.fromFloats(new float[]{0.5f}, new long[]{1});
        when(preprocessor.process(any(BufferedImage.class))).thenReturn(inputTensor);

        // Raw output: [3, 4] -> L2 norm = 5, normalized = [0.6, 0.8]
        Tensor outputTensor = Tensor.fromFloats(new float[]{3.0f, 4.0f}, new long[]{1, 2});
        when(session.run(any())).thenReturn(Map.of("image_embeds", outputTensor));

        ClipImageEncoder encoder = ClipImageEncoder.builder()
                .session(session)
                .preprocessor(preprocessor)
                .build();

        BufferedImage image = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        float[] embedding = encoder.encode(image);

        assertEquals(2, embedding.length);
        assertEquals(0.6f, embedding[0], 1e-5f);
        assertEquals(0.8f, embedding[1], 1e-5f);

        // Verify L2 norm ≈ 1.0
        float norm = 0f;
        for (float v : embedding) {
            norm += v * v;
        }
        assertEquals(1.0f, (float) Math.sqrt(norm), 1e-5f);
    }

    @Test
    void encodeBatch_returnsOneEmbeddingPerImage() {
        InferenceSession session = mock(InferenceSession.class);
        @SuppressWarnings("unchecked")
        Preprocessor<BufferedImage, Tensor> preprocessor = mock(Preprocessor.class);

        Tensor inputTensor = Tensor.fromFloats(new float[]{0.5f}, new long[]{1});
        when(preprocessor.process(any(BufferedImage.class))).thenReturn(inputTensor);

        Tensor outputTensor = Tensor.fromFloats(new float[]{1.0f, 0.0f}, new long[]{1, 2});
        when(session.run(any())).thenReturn(Map.of("image_embeds", outputTensor));

        ClipImageEncoder encoder = ClipImageEncoder.builder()
                .session(session)
                .preprocessor(preprocessor)
                .build();

        List<BufferedImage> images = List.of(
                new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB),
                new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB));
        List<float[]> results = encoder.encodeBatch(images);

        assertEquals(2, results.size());
        for (float[] emb : results) {
            assertEquals(2, emb.length);
        }
    }

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThrows(ModelSourceException.class, () ->
                ClipImageEncoder.builder()
                        .modelSource(badSource)
                        .build());
    }

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);
        @SuppressWarnings("unchecked")
        Preprocessor<BufferedImage, Tensor> preprocessor = mock(Preprocessor.class);

        ClipImageEncoder encoder = ClipImageEncoder.builder()
                .session(session)
                .preprocessor(preprocessor)
                .build();

        encoder.close();

        verify(session).close();
    }

    @Test
    void clipNormalization_usesCorrectMeanAndStd() {
        assertArrayEquals(
                new float[]{0.48145466f, 0.4578275f, 0.40821073f},
                ClipImageEncoder.CLIP_MEAN, 1e-7f);
        assertArrayEquals(
                new float[]{0.26862954f, 0.26130258f, 0.27577711f},
                ClipImageEncoder.CLIP_STD, 1e-7f);
    }
}
