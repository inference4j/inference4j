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
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ClipTextEncoderTest {

    // The fixture model counts the unmasked tokens with ids 0, 1 and 2 into a 3-value raw embedding.

    @TempDir
    Path dir;

    @Test
    void encode_returnsL2NormalizedEmbedding() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString())).thenReturn(
                new EncodedInput(new long[]{0, 0, 0, 1, 1, 1, 1}, ones(7), new long[7]));

        try (ClipTextEncoder encoder = encoder(tokenizer)) {
            float[] embedding = encoder.encode("a photo of a cat");

            // Raw output: [3, 4, 0] -> L2 norm = 5, normalized = [0.6, 0.8, 0]
            assertThat(embedding).containsExactly(new float[]{0.6f, 0.8f, 0f}, within(1e-5f));
        }
    }

    @Test
    void encode_passesInputIdsAndAttentionMask() {
        // Unmasked, the ids would count to [1, 2, 0]; with the last token masked they count to [1, 1, 0]
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("hello")).thenReturn(
                new EncodedInput(new long[]{0, 1, 1}, new long[]{1, 1, 0}, new long[]{0, 0, 0}));

        try (ClipTextEncoder encoder = encoder(tokenizer)) {
            float half = (float) Math.sqrt(0.5);
            assertThat(encoder.encode("hello")).containsExactly(new float[]{half, half, 0f}, within(1e-5f));
        }
    }

    @Test
    void encodeBatch_returnsOneEmbeddingPerText() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("cat")).thenReturn(new EncodedInput(new long[]{0, 4}, ones(2), new long[2]));
        when(tokenizer.encode("dog")).thenReturn(new EncodedInput(new long[]{1, 4}, ones(2), new long[2]));
        when(tokenizer.encode("bird")).thenReturn(new EncodedInput(new long[]{2, 4}, ones(2), new long[2]));

        try (ClipTextEncoder encoder = encoder(tokenizer)) {
            List<float[]> results = encoder.encodeBatch(List.of("cat", "dog", "bird"));

            assertThat(results).containsExactly(
                    new float[]{1f, 0f, 0f}, new float[]{0f, 1f, 0f}, new float[]{0f, 0f, 1f});
        }
    }

    @Test
    void builder_missingTokenizer_throws() {
        TestModels.install(dir, "clip-text", "text_model.onnx");

        assertThatThrownBy(() ->
                ClipTextEncoder.builder()
                        .modelSource(TestModels.source(dir))
                        .build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("vocab.json");
    }

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                ClipTextEncoder.builder()
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void closedEncoderRejectsEncode() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString())).thenReturn(new EncodedInput(new long[]{0}, ones(1), new long[1]));
        ClipTextEncoder encoder = encoder(tokenizer);

        encoder.close();

        assertThatThrownBy(() -> encoder.encode("hello")).isInstanceOf(IllegalStateException.class);
    }

    private ClipTextEncoder encoder(Tokenizer tokenizer) {
        TestModels.install(dir, "clip-text", "text_model.onnx");
        return ClipTextEncoder.builder()
                .modelSource(TestModels.source(dir))
                .tokenizer(tokenizer)
                .build();
    }

    private static long[] ones(int n) {
        long[] ones = new long[n];
        Arrays.fill(ones, 1L);
        return ones;
    }
}
