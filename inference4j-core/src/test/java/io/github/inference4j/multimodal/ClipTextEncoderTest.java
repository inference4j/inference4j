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
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ClipTextEncoderTest {

    @Test
    void encode_returnsL2NormalizedEmbedding() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(tokenizer.encode(anyString())).thenReturn(
                new EncodedInput(new long[]{1, 2, 3}, new long[]{1, 1, 1}, new long[]{0, 0, 0}));

        // Raw output: [3, 4] -> L2 norm = 5, normalized = [0.6, 0.8]
        Tensor outputTensor = Tensor.fromFloats(new float[]{3.0f, 4.0f}, new long[]{1, 2});
        when(session.run(any())).thenReturn(Map.of("text_embeds", outputTensor));

        io.github.inference4j.multimodal.ClipTextEncoder encoder = ClipTextEncoder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        float[] embedding = encoder.encode("a photo of a cat");

        assertEquals(2, embedding.length);
        assertEquals(0.6f, embedding[0], 1e-5f);
        assertEquals(0.8f, embedding[1], 1e-5f);
    }

    @Test
    void encode_passesInputIdsAndAttentionMask() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        long[] inputIds = {49406, 320, 49407};
        long[] attentionMask = {1, 1, 1};
        when(tokenizer.encode("hello")).thenReturn(
                new EncodedInput(inputIds, attentionMask, new long[]{0, 0, 0}));

        Tensor outputTensor = Tensor.fromFloats(new float[]{1.0f, 0.0f}, new long[]{1, 2});
        when(session.run(any())).thenReturn(Map.of("text_embeds", outputTensor));

        ClipTextEncoder encoder = ClipTextEncoder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        encoder.encode("hello");

        verify(session).run(argThat(inputs -> {
            assertTrue(inputs.containsKey("input_ids"));
            assertTrue(inputs.containsKey("attention_mask"));
            assertArrayEquals(inputIds, inputs.get("input_ids").toLongs());
            assertArrayEquals(attentionMask, inputs.get("attention_mask").toLongs());
            return true;
        }));
    }

    @Test
    void encodeBatch_returnsOneEmbeddingPerText() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(tokenizer.encode(anyString())).thenReturn(
                new EncodedInput(new long[]{1, 2}, new long[]{1, 1}, new long[]{0, 0}));

        Tensor outputTensor = Tensor.fromFloats(new float[]{1.0f, 0.0f}, new long[]{1, 2});
        when(session.run(any())).thenReturn(Map.of("text_embeds", outputTensor));

        ClipTextEncoder encoder = ClipTextEncoder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        List<float[]> results = encoder.encodeBatch(List.of("cat", "dog", "bird"));
        assertEquals(3, results.size());
    }

    @Test
    void builder_missingTokenizer_throws() {
        InferenceSession session = mock(InferenceSession.class);

        assertThrows(IllegalStateException.class, () ->
                ClipTextEncoder.builder()
                        .session(session)
                        .build());
    }

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThrows(ModelSourceException.class, () ->
                ClipTextEncoder.builder()
                        .modelSource(badSource)
                        .build());
    }

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        ClipTextEncoder encoder = ClipTextEncoder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        encoder.close();

        verify(session).close();
    }
}
