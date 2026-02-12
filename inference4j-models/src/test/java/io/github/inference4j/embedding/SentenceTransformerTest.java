package io.github.inference4j.embedding;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

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
}
