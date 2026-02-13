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

package io.github.inference4j.embedding;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class SentenceTransformerTest {

    @Test
    void meanPooling_averagesAcrossTokensWeightedByMask() {
        // Shape: [1, 3, 4] — 1 batch, 3 tokens, 4 hidden dims
        float[] flatOutput = {
                1f, 2f, 3f, 4f,    // token 0
                5f, 6f, 7f, 8f,    // token 1
                9f, 10f, 11f, 12f  // token 2
        };
        long[] shape = {1, 3, 4};
        long[] attentionMask = {1, 1, 0}; // only first 2 tokens are real

        float[] result = SentenceTransformer.applyPooling(flatOutput, shape, attentionMask, PoolingStrategy.MEAN);

        // Mean of token 0 and token 1: (1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2
        assertArrayEquals(new float[]{3f, 4f, 5f, 6f}, result, 0.001f);
    }

    @Test
    void clsPooling_returnsFirstTokenEmbedding() {
        float[] flatOutput = {
                1f, 2f, 3f, 4f,    // token 0 (CLS)
                5f, 6f, 7f, 8f,    // token 1
                9f, 10f, 11f, 12f  // token 2
        };
        long[] shape = {1, 3, 4};
        long[] attentionMask = {1, 1, 1};

        float[] result = SentenceTransformer.applyPooling(flatOutput, shape, attentionMask, PoolingStrategy.CLS);

        assertArrayEquals(new float[]{1f, 2f, 3f, 4f}, result);
    }

    @Test
    void maxPooling_takesElementWiseMaxAcrossMaskedTokens() {
        float[] flatOutput = {
                1f, 6f, 3f, 8f,    // token 0
                5f, 2f, 7f, 4f,    // token 1
                9f, 10f, 11f, 12f  // token 2 — masked out
        };
        long[] shape = {1, 3, 4};
        long[] attentionMask = {1, 1, 0};

        float[] result = SentenceTransformer.applyPooling(flatOutput, shape, attentionMask, PoolingStrategy.MAX);

        // Max of token 0 and 1: max(1,5), max(6,2), max(3,7), max(8,4)
        assertArrayEquals(new float[]{5f, 6f, 7f, 8f}, result);
    }

    @Test
    void meanPooling_allTokensMasked_returnsZeros() {
        float[] flatOutput = {1f, 2f, 3f, 4f};
        long[] shape = {1, 1, 4};
        long[] attentionMask = {0};

        float[] result = SentenceTransformer.applyPooling(flatOutput, shape, attentionMask, PoolingStrategy.MEAN);

        assertArrayEquals(new float[]{0f, 0f, 0f, 0f}, result);
    }

    // --- Builder validation ---

    @Test
    void builder_missingSession_throws() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThrows(IllegalStateException.class, () ->
                SentenceTransformer.builder()
                        .tokenizer(tokenizer)
                        .build());
    }

    @Test
    void builder_missingTokenizer_throws() {
        InferenceSession session = mock(InferenceSession.class);
        assertThrows(IllegalStateException.class, () ->
                SentenceTransformer.builder()
                        .session(session)
                        .build());
    }

    // --- Inference flow ---

    @Test
    void encode_withTokenTypeIds_returnsEmbedding() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask", "token_type_ids"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 2023, 102}, new long[]{1, 1, 1}, new long[]{0, 0, 0}));

        // Shape [1, 3, 4]: 3 tokens, 4 hidden dims
        float[] output = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f};
        when(session.run(any())).thenReturn(
                Map.of("output", Tensor.fromFloats(output, new long[]{1, 3, 4})));

        SentenceTransformer model = SentenceTransformer.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build(); // default MEAN pooling

        float[] embedding = model.encode("hello world");

        // MEAN of all 3 tokens (all mask=1): (1+5+9)/3, (2+6+10)/3, (3+7+11)/3, (4+8+12)/3
        assertEquals(4, embedding.length);
        assertEquals(5f, embedding[0], 0.001f);
        assertEquals(6f, embedding[1], 0.001f);
        assertEquals(7f, embedding[2], 0.001f);
        assertEquals(8f, embedding[3], 0.001f);

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(session).run(captor.capture());
        assertTrue(captor.getValue().containsKey("token_type_ids"));
    }

    @Test
    void encode_withoutTokenTypeIds_excludesFromInputs() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 2023, 102}, new long[]{1, 1, 1}, new long[]{0, 0, 0}));

        float[] output = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f};
        when(session.run(any())).thenReturn(
                Map.of("output", Tensor.fromFloats(output, new long[]{1, 3, 4})));

        SentenceTransformer model = SentenceTransformer.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        model.encode("hello world");

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(session).run(captor.capture());
        assertFalse(captor.getValue().containsKey("token_type_ids"));
    }

    @Test
    void encodeBatch_returnsEmbeddingsForAllTexts() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 2023, 102}, new long[]{1, 1, 1}, new long[]{0, 0, 0}));

        float[] output1 = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f};
        float[] output2 = {12f, 11f, 10f, 9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f};
        when(session.run(any()))
                .thenReturn(Map.of("output", Tensor.fromFloats(output1, new long[]{1, 3, 4})))
                .thenReturn(Map.of("output", Tensor.fromFloats(output2, new long[]{1, 3, 4})));

        SentenceTransformer model = SentenceTransformer.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        List<float[]> results = model.encodeBatch(List.of("text1", "text2"));

        assertEquals(2, results.size());
        assertEquals(4, results.get(0).length);
        assertEquals(4, results.get(1).length);
        // First result: MEAN of [1,5,9],[2,6,10],[3,7,11],[4,8,12] → [5,6,7,8]
        assertEquals(5f, results.get(0)[0], 0.001f);
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        SentenceTransformer model = SentenceTransformer.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        model.close();

        verify(session).close();
    }
}
