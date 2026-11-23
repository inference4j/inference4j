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
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class SentenceTransformerEmbedderTest {

    @TempDir
    Path dir;

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
    void builderWithoutModelIdThrows() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThatThrownBy(() ->
                SentenceTransformerEmbedder.builder()
                        .tokenizer(tokenizer)
                        .build())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("modelId");
    }

    @Test
    void builder_missingTokenizer_throws() {
        ModelSource source = TestModels.model(dir, "tokens-onehot");
        assertThatThrownBy(() ->
                SentenceTransformerEmbedder.builder()
                        .modelId("acme/tiny")
                        .modelSource(source)
                        .build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("vocab.txt");
    }

    // --- Inference flow ---
    // The fixture model's hidden state for a token is 10 at the index of its id among 16 dimensions, so
    // mean pooling over ids (0, 1, 1) gives [10/3, 20/3, 0, ...].

    @Test
    void encode_withTokenTypeIds_returnsEmbedding() {
        // A model with token_type_ids requires them; one without would reject them
        Tokenizer tokenizer = tokenizerReturning(0, 1, 1);

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot-types", tokenizer).build()) {
            float[] embedding = model.encode("hello world"); // default MEAN pooling

            assertThat(embedding).hasSize(16);
            assertThat(embedding).containsExactly(meanOf011(), within(1e-5f));
        }
    }

    @Test
    void encode_withoutTokenTypeIds_excludesFromInputs() {
        Tokenizer tokenizer = tokenizerReturning(0, 1, 1);

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).build()) {
            assertThat(model.encode("hello world")).containsExactly(meanOf011(), within(1e-5f));
        }
    }

    @Test
    void encodeBatch_returnsEmbeddingsForAllTexts() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(eq("text1"), anyInt())).thenReturn(encoded(0, 1, 1));
        when(tokenizer.encode(eq("text2"), anyInt())).thenReturn(encoded(3, 3, 2));

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).build()) {
            List<float[]> results = model.encodeBatch(List.of("text1", "text2"));

            assertThat(results).hasSize(2);
            assertThat(results.get(0)).containsExactly(meanOf011(), within(1e-5f));
            float[] second = new float[16];
            second[2] = 10f / 3;
            second[3] = 20f / 3;
            assertThat(results.get(1)).containsExactly(second, within(1e-5f));
        }
    }

    // --- Builder setters ---

    @Test
    void builder_poolingStrategy_appliesCLS() {
        Tokenizer tokenizer = tokenizerReturning(0, 1, 1);

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer)
                .poolingStrategy(PoolingStrategy.CLS)
                .build()) {
            float[] embedding = model.encode("hello");

            // CLS returns the first token: 10 at index 0
            float[] expected = new float[16];
            expected[0] = 10f;
            assertThat(embedding).isEqualTo(expected);
        }
    }

    @Test
    void builder_maxLength_passedToTokenizer() {
        Tokenizer tokenizer = tokenizerReturning(1, 2);

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).maxLength(64).build()) {
            model.encode("hello");
        }

        verify(tokenizer).encode("hello", 64);
    }

    // --- Normalize ---

    @Test
    void normalize_producesUnitVector() {
        Tokenizer tokenizer = tokenizerReturning(0, 1, 1);

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).normalize().build()) {
            float[] embedding = model.encode("hello");

            // [1, 2, 0, ...] / sqrt(5)
            assertThat(embedding[0]).isCloseTo((float) (1 / Math.sqrt(5)), within(1e-5f));
            assertThat(embedding[1]).isCloseTo((float) (2 / Math.sqrt(5)), within(1e-5f));
            float norm = 0f;
            for (float v : embedding) {
                norm += v * v;
            }
            assertThat((double) Math.sqrt(norm)).isCloseTo(1.0, within(1e-4));
        }
    }

    @Test
    void withoutNormalize_doesNotNormalize() {
        Tokenizer tokenizer = tokenizerReturning(0, 1, 1);

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).build()) {
            // Without normalize, result is raw MEAN pooling (not unit norm)
            assertThat(model.encode("hello")[0]).isCloseTo(10f / 3, within(0.001f));
        }
    }

    // --- Text prefix ---

    @Test
    void textPrefix_prependedBeforeEncoding() {
        Tokenizer tokenizer = tokenizerReturning(1, 2);

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer)
                .textPrefix("query: ")
                .build()) {
            model.encode("hello");
        }

        verify(tokenizer).encode("query: hello", 512);
    }

    @Test
    void withoutTextPrefix_encodesAsIs() {
        Tokenizer tokenizer = tokenizerReturning(1, 2);

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).build()) {
            model.encode("hello");
        }

        verify(tokenizer).encode("hello", 512);
    }

    // --- Per-call prefix ---

    @Test
    void perCallPrefixReplacesTheBuilderPrefix() {
        Tokenizer tokenizer = tokenizerReturning(1, 2);
        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).textPrefix("query: ").build()) {
            model.encode("hello", "passage: ");
        }

        verify(tokenizer).encode("passage: hello", 512);
        verify(tokenizer, never()).encode(eq("query: hello"), anyInt());
    }

    @Test
    void nullPerCallPrefixMeansNoPrefix() {
        Tokenizer tokenizer = tokenizerReturning(1, 2);
        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).textPrefix("query: ").build()) {
            model.encode("hello", null);
        }

        verify(tokenizer).encode("hello", 512);
    }

    @Test
    void encodeBatchAppliesThePerCallPrefixToEveryTextInOrder() {
        Tokenizer tokenizer = tokenizerReturning(1, 2);
        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).build()) {
            List<float[]> embeddings = model.encodeBatch(List.of("a", "b"), "search_document: ");

            assertThat(embeddings).hasSize(2);
        }

        var order = inOrder(tokenizer);
        order.verify(tokenizer).encode("search_document: a", 512);
        order.verify(tokenizer).encode("search_document: b", 512);
    }

    @Test
    void runStillAppliesTheBuilderPrefix() {
        Tokenizer tokenizer = tokenizerReturning(1, 2);
        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer).textPrefix("query: ").build()) {
            model.run("hello");
        }

        verify(tokenizer).encode("query: hello", 512);
    }

    // maxLength 8 with a 2-token prefix leaves 4 content tokens per window; 10 content tokens with
    // stride 2 give windows starting at 0, 2, 4 and 6. Ids: [CLS] 14, [SEP] 15, prefix 12 and 13, content 1..10.

    @Test
    void strideRepeatsThePerCallPrefixInEveryWindow() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("p: long text", Integer.MAX_VALUE))
                .thenReturn(encoded(14, 12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15));
        when(tokenizer.encode("p: ", Integer.MAX_VALUE)).thenReturn(encoded(14, 12, 13, 15));

        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer)
                .maxLength(8).stride(2).build()) {
            float[] embedding = model.encode("long text", "p: ");

            // Each window of 8 tokens pools to 10/8 at its ids. The windows are averaged weighted by the tokens
            // each owns (3, 2, 2 and 3 of the 10), so [CLS], [SEP] and the prefix (in every window) keep 10/8,
            // and a content token keeps 10/8 x the owned share of the windows it appears in
            float window = 10f / 8;
            float[] expected = new float[16];
            float[] shareOfWindowsWithToken = {0, .3f, .3f, .5f, .5f, .4f, .4f, .5f, .5f, .3f, .3f};
            for (int id = 1; id <= 10; id++) {
                expected[id] = window * shareOfWindowsWithToken[id];
            }
            for (int id = 12; id <= 15; id++) {
                expected[id] = window;
            }
            assertThat(embedding).containsExactly(expected, within(1e-5f));
        }
    }

    @Test
    void strideRejectsAPerCallPrefixThatLeavesNoRoomBeyondTheStride() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("long prefix: long text", Integer.MAX_VALUE))
                .thenReturn(encoded(14, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 15));
        when(tokenizer.encode("long prefix: ", Integer.MAX_VALUE)).thenReturn(encoded(14, 11, 12, 13, 0, 15));
        try (SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizer)
                .maxLength(8).stride(2).build()) {
            assertThatThrownBy(() -> model.encode("long text", "long prefix: "))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("stride 2");
        }
    }

    // --- Close ---

    @Test
    void closedEmbedderRejectsEncode() {
        SentenceTransformerEmbedder model = embedder("tokens-onehot", tokenizerReturning(0, 1, 1)).build();

        model.close();

        assertThatThrownBy(() -> model.encode("hello")).isInstanceOf(IllegalStateException.class);
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

    /** Mean pooling over ids (0, 1, 1): [10/3, 20/3, 0, ...] in 16 dimensions. */
    private SentenceTransformerEmbedder.Builder meanPooledBuilder() {
        return embedder("tokens-onehot", tokenizerReturning(0, 1, 1));
    }

    @Test
    void dimensionsKeepsTheLeadingValues() {
        try (var model = meanPooledBuilder().dimensions(2).build()) {
            assertThat(model.encode("hello")).containsExactly(new float[]{10f / 3, 20f / 3}, within(1e-5f));
        }
    }

    @Test
    void dimensionsRenormalizesAfterTruncation() {
        try (var model = meanPooledBuilder().dimensions(2).normalize().build()) {
            float norm = (float) Math.sqrt(5);
            assertThat(model.encode("hello")).containsExactly(new float[]{1f / norm, 2f / norm}, within(1e-6f));
        }
    }

    @Test
    void dimensionsLargerThanTheEmbeddingFails() {
        try (var model = meanPooledBuilder().dimensions(17).build()) {
            assertThatThrownBy(() -> model.encode("hello"))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("embedding size (16)");
        }
    }

    @Test
    void dimensionsMustBePositive() {
        assertThatThrownBy(() -> SentenceTransformerEmbedder.builder().dimensions(0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    // --- Helpers ---

    private SentenceTransformerEmbedder.Builder embedder(String fixture, Tokenizer tokenizer) {
        return SentenceTransformerEmbedder.builder()
                .modelId("acme/tiny")
                .modelSource(TestModels.model(dir, fixture))
                .tokenizer(tokenizer);
    }

    private static Tokenizer tokenizerReturning(long... ids) {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(encoded(ids));
        return tokenizer;
    }

    private static EncodedInput encoded(long... ids) {
        long[] mask = new long[ids.length];
        Arrays.fill(mask, 1L);
        return new EncodedInput(ids, mask, new long[ids.length]);
    }

    private static float[] meanOf011() {
        float[] expected = new float[16];
        expected[0] = 10f / 3;
        expected[1] = 20f / 3;
        return expected;
    }
}
