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
import java.util.ArrayList;
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

    // --- Normalize ---

    @Test
    void normalize_producesUnitVector() {
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
                .normalize()
                .build();

        float[] embedding = model.encode("hello");

        // Verify unit norm
        float norm = 0f;
        for (float v : embedding) {
            norm += v * v;
        }
        assertThat((double) Math.sqrt(norm)).isCloseTo(1.0, within(1e-4));
    }

    @Test
    void withoutNormalize_doesNotNormalize() {
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

        float[] embedding = model.encode("hello");

        // Without normalize, result is raw MEAN pooling (not unit norm)
        assertThat(embedding[0]).isCloseTo(5f, within(0.001f));
    }

    // --- Text prefix ---

    @Test
    void textPrefix_prependedBeforeEncoding() {
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
                .textPrefix("query: ")
                .build();

        model.encode("hello");

        verify(tokenizer).encode("query: hello", 512);
    }

    @Test
    void withoutTextPrefix_encodesAsIs() {
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
                .build();

        model.encode("hello");

        verify(tokenizer).encode("hello", 512);
    }

    // --- Per-call prefix ---

    private static EncodedInput encoded(long... ids) {
        long[] mask = new long[ids.length];
        java.util.Arrays.fill(mask, 1L);
        return new EncodedInput(ids, mask, new long[ids.length]);
    }

    private static InferenceSession shortSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(session.run(any())).thenReturn(
                Map.of("output", Tensor.fromFloats(new float[]{1f, 2f, 3f, 4f}, new long[]{1, 2, 2})));
        return session;
    }

    private static Tokenizer shortTokenizer() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(encoded(101, 102));
        return tokenizer;
    }

    @Test
    void perCallPrefixReplacesTheBuilderPrefix() {
        Tokenizer tokenizer = shortTokenizer();
        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(shortSession()).tokenizer(tokenizer).textPrefix("query: ").build();

        model.encode("hello", "passage: ");

        verify(tokenizer).encode("passage: hello", 512);
        verify(tokenizer, never()).encode(eq("query: hello"), anyInt());
    }

    @Test
    void nullPerCallPrefixMeansNoPrefix() {
        Tokenizer tokenizer = shortTokenizer();
        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(shortSession()).tokenizer(tokenizer).textPrefix("query: ").build();

        model.encode("hello", null);

        verify(tokenizer).encode("hello", 512);
    }

    @Test
    void encodeBatchAppliesThePerCallPrefixToEveryTextInOrder() {
        Tokenizer tokenizer = shortTokenizer();
        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(shortSession()).tokenizer(tokenizer).build();

        List<float[]> embeddings = model.encodeBatch(List.of("a", "b"), "search_document: ");

        assertThat(embeddings).hasSize(2);
        var order = inOrder(tokenizer);
        order.verify(tokenizer).encode("search_document: a", 512);
        order.verify(tokenizer).encode("search_document: b", 512);
    }

    @Test
    void runStillAppliesTheBuilderPrefix() {
        Tokenizer tokenizer = shortTokenizer();
        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(shortSession()).tokenizer(tokenizer).textPrefix("query: ").build();

        model.run("hello");

        verify(tokenizer).encode("query: hello", 512);
    }

    // maxLength 8 with a 2-token prefix leaves 4 content tokens per window; 10 content tokens with
    // stride 2 give windows starting at 0, 2, 4 and 6.

    @Test
    @SuppressWarnings("unchecked")
    void strideRepeatsThePerCallPrefixInEveryWindow() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(session.run(any())).thenReturn(
                Map.of("output", Tensor.fromFloats(new float[8], new long[]{1, 8, 1})));
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("p: long text", Integer.MAX_VALUE))
                .thenReturn(encoded(101, 50, 51, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 102));
        when(tokenizer.encode("p: ", Integer.MAX_VALUE)).thenReturn(encoded(101, 50, 51, 102));
        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(session).tokenizer(tokenizer).maxLength(8).stride(2).build();

        model.encode("long text", "p: ");

        ArgumentCaptor<Map<String, Tensor>> inputs = ArgumentCaptor.forClass(Map.class);
        verify(session, times(4)).run(inputs.capture());
        List<long[]> windows = new ArrayList<>();
        inputs.getAllValues().forEach(in -> windows.add(in.get("input_ids").toLongs()));
        assertThat(windows).containsExactly(
                new long[]{101, 50, 51, 1, 2, 3, 4, 102},
                new long[]{101, 50, 51, 3, 4, 5, 6, 102},
                new long[]{101, 50, 51, 5, 6, 7, 8, 102},
                new long[]{101, 50, 51, 7, 8, 9, 10, 102});
    }

    @Test
    void strideRejectsAPerCallPrefixThatLeavesNoRoomBeyondTheStride() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("long prefix: long text", Integer.MAX_VALUE))
                .thenReturn(encoded(101, 50, 51, 52, 53, 1, 2, 3, 4, 5, 6, 102));
        when(tokenizer.encode("long prefix: ", Integer.MAX_VALUE)).thenReturn(encoded(101, 50, 51, 52, 53, 102));
        SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .session(shortSession()).tokenizer(tokenizer).maxLength(8).stride(2).build();

        assertThatThrownBy(() -> model.encode("long text", "long prefix: "))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("stride 2");
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

    @Test
    void weightedAverageWeighsEachVectorByItsTokens() {
        float[] average = SentenceTransformerEmbedder.weightedAverage(
                List.of(new float[]{1f, 0f}, new float[]{0f, 1f}),
                List.of(3, 1));

        assertThat(average[0]).isCloseTo(0.75f, within(1e-6f));
        assertThat(average[1]).isCloseTo(0.25f, within(1e-6f));
    }

    // --- Matryoshka dimensions ---

    private static SentenceTransformerEmbedder.Builder meanPooledBuilder() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 2023, 102}, new long[]{1, 1, 1}, new long[]{0, 0, 0}));
        // Mean over the three tokens is [5, 6, 7, 8]
        float[] output = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f};
        when(session.run(any())).thenReturn(
                Map.of("output", Tensor.fromFloats(output, new long[]{1, 3, 4})));
        return SentenceTransformerEmbedder.builder().session(session).tokenizer(tokenizer);
    }

    @Test
    void dimensionsKeepsTheLeadingValues() {
        try (var model = meanPooledBuilder().dimensions(2).build()) {
            assertThat(model.encode("hello")).containsExactly(5f, 6f);
        }
    }

    @Test
    void dimensionsRenormalizesAfterTruncation() {
        try (var model = meanPooledBuilder().dimensions(2).normalize().build()) {
            float norm = (float) Math.sqrt(61);
            assertThat(model.encode("hello")).containsExactly(new float[]{5f / norm, 6f / norm}, within(1e-6f));
        }
    }

    @Test
    void dimensionsLargerThanTheEmbeddingFails() {
        try (var model = meanPooledBuilder().dimensions(5).build()) {
            assertThatThrownBy(() -> model.encode("hello"))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("embedding size (4)");
        }
    }

    @Test
    void dimensionsMustBePositive() {
        assertThatThrownBy(() -> SentenceTransformerEmbedder.builder().dimensions(0))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
