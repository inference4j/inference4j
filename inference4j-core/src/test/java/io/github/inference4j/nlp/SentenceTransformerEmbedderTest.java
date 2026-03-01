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

package io.github.inference4j.nlp;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class SentenceTransformerEmbedderTest {

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

        float[] result = io.github.inference4j.nlp.SentenceTransformerEmbedder.applyPooling(flatOutput, shape, attentionMask, PoolingStrategy.MEAN);

        // Mean of token 0 and token 1: (1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2
        assertThat(result).containsExactly(new float[]{3f, 4f, 5f, 6f}, within(0.001f));
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

        float[] result = SentenceTransformerEmbedder.applyPooling(flatOutput, shape, attentionMask, PoolingStrategy.CLS);

        assertThat(result).isEqualTo(new float[]{1f, 2f, 3f, 4f});
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

        float[] result = SentenceTransformerEmbedder.applyPooling(flatOutput, shape, attentionMask, PoolingStrategy.MAX);

        // Max of token 0 and 1: max(1,5), max(6,2), max(3,7), max(8,4)
        assertThat(result).isEqualTo(new float[]{5f, 6f, 7f, 8f});
    }

    @Test
    void meanPooling_allTokensMasked_returnsZeros() {
        float[] flatOutput = {1f, 2f, 3f, 4f};
        long[] shape = {1, 1, 4};
        long[] attentionMask = {0};

        float[] result = SentenceTransformerEmbedder.applyPooling(flatOutput, shape, attentionMask, PoolingStrategy.MEAN);

        assertThat(result).isEqualTo(new float[]{0f, 0f, 0f, 0f});
    }

    // --- Builder validation ---

    @Test
    void builder_missingSession_throws() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThatThrownBy(() ->
                SentenceTransformerEmbedder.builder()
                        .tokenizer(tokenizer)
                        .build())
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void builder_missingTokenizer_throws() {
        InferenceSession session = mock(InferenceSession.class);
        assertThatThrownBy(() ->
                SentenceTransformerEmbedder.builder()
                        .session(session)
                        .build())
                .isInstanceOf(IllegalStateException.class);
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

        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build(); // default MEAN pooling

        float[] embedding = model.encode("hello world");

        // MEAN of all 3 tokens (all mask=1): (1+5+9)/3, (2+6+10)/3, (3+7+11)/3, (4+8+12)/3
        assertThat(embedding).hasSize(4);
        assertThat(embedding[0]).isCloseTo(5f, within(0.001f));
        assertThat(embedding[1]).isCloseTo(6f, within(0.001f));
        assertThat(embedding[2]).isCloseTo(7f, within(0.001f));
        assertThat(embedding[3]).isCloseTo(8f, within(0.001f));

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(session).run(captor.capture());
        assertThat(captor.getValue()).containsKey("token_type_ids");
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

        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        model.encode("hello world");

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(session).run(captor.capture());
        assertThat(captor.getValue()).doesNotContainKey("token_type_ids");
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

        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        List<float[]> results = model.encodeBatch(List.of("text1", "text2"));

        assertThat(results).hasSize(2);
        assertThat(results.get(0)).hasSize(4);
        assertThat(results.get(1)).hasSize(4);
        // First result: MEAN of [1,5,9],[2,6,10],[3,7,11],[4,8,12] → [5,6,7,8]
        assertThat(results.get(0)[0]).isCloseTo(5f, within(0.001f));
    }

    // --- Builder setters ---

    @Test
    void builder_poolingStrategy_appliesCLS() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 2023, 102}, new long[]{1, 1, 1}, new long[]{0, 0, 0}));

        float[] output = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f};
        when(session.run(any())).thenReturn(
                Map.of("output", Tensor.fromFloats(output, new long[]{1, 3, 4})));

        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .poolingStrategy(PoolingStrategy.CLS)
                .build();

        float[] embedding = model.encode("hello");

        // CLS returns first token: [1, 2, 3, 4]
        assertThat(embedding).isEqualTo(new float[]{1f, 2f, 3f, 4f});
    }

    @Test
    void builder_maxLength_passedToTokenizer() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 102}, new long[]{1, 1}, new long[]{0, 0}));

        float[] output = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        when(session.run(any())).thenReturn(
                Map.of("output", Tensor.fromFloats(output, new long[]{1, 2, 4})));

        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .maxLength(64)
                .build();

        model.encode("hello");

        verify(tokenizer).encode("hello", 64);
    }

    @Test
    void builder_modelId_usedWithModelSource() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        // Verify modelId and modelSource setters don't throw
        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .modelId("custom/model")
                .modelSource(id -> Path.of("/tmp"))
                .build();

        assertThat(model).isNotNull();
    }

    @Test
    void builder_sessionOptions_accepted() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .sessionOptions(opts -> { })
                .build();

        assertThat(model).isNotNull();
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        model.close();

        verify(session).close();
    }
}
