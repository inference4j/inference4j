package io.github.inference4j;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MathOpsTest {

    @Test
    void softmax_sumsToOne() {
        float[] logits = {1.0f, 2.0f, 3.0f};
        float[] result = MathOps.softmax(logits);

        float sum = 0f;
        for (float v : result) {
            sum += v;
        }
        assertEquals(1.0f, sum, 1e-5f);
    }

    @Test
    void softmax_largestInputGetsHighestProbability() {
        float[] logits = {1.0f, 5.0f, 2.0f};
        float[] result = MathOps.softmax(logits);

        assertTrue(result[1] > result[0]);
        assertTrue(result[1] > result[2]);
    }

    @Test
    void softmax_uniformInputsGiveEqualProbabilities() {
        float[] logits = {3.0f, 3.0f, 3.0f};
        float[] result = MathOps.softmax(logits);

        assertEquals(result[0], result[1], 1e-6f);
        assertEquals(result[1], result[2], 1e-6f);
        assertEquals(1.0f / 3, result[0], 1e-5f);
    }

    @Test
    void softmax_numericallyStableWithLargeValues() {
        float[] logits = {1000f, 1001f, 1002f};
        float[] result = MathOps.softmax(logits);

        float sum = 0f;
        for (float v : result) {
            assertFalse(Float.isNaN(v), "softmax produced NaN");
            assertFalse(Float.isInfinite(v), "softmax produced Inf");
            sum += v;
        }
        assertEquals(1.0f, sum, 1e-5f);
        assertTrue(result[2] > result[1]);
        assertTrue(result[1] > result[0]);
    }

    @Test
    void softmax_singleElement() {
        float[] result = MathOps.softmax(new float[]{42f});
        assertEquals(1, result.length);
        assertEquals(1.0f, result[0], 1e-6f);
    }

    @Test
    void softmax_negativeValues() {
        float[] logits = {-1f, -2f, -3f};
        float[] result = MathOps.softmax(logits);

        float sum = 0f;
        for (float v : result) {
            assertTrue(v > 0f);
            sum += v;
        }
        assertEquals(1.0f, sum, 1e-5f);
        assertTrue(result[0] > result[1]);
        assertTrue(result[1] > result[2]);
    }

    @Test
    void sigmoid_outputsBetweenZeroAndOne() {
        float[] values = {-5f, -1f, 0f, 1f, 5f};
        float[] result = MathOps.sigmoid(values);

        for (float v : result) {
            assertTrue(v > 0f && v < 1f);
        }
    }

    @Test
    void sigmoid_zeroInputGivesHalf() {
        float[] result = MathOps.sigmoid(new float[]{0f});
        assertEquals(0.5f, result[0], 1e-6f);
    }

    @Test
    void sigmoid_symmetricAroundZero() {
        float[] result = MathOps.sigmoid(new float[]{-2f, 2f});
        assertEquals(1.0f, result[0] + result[1], 1e-5f);
    }

    @Test
    void sigmoid_largePositiveCloseToOne() {
        float[] result = MathOps.sigmoid(new float[]{100f});
        assertEquals(1.0f, result[0], 1e-5f);
    }

    @Test
    void sigmoid_largeNegativeCloseToZero() {
        float[] result = MathOps.sigmoid(new float[]{-100f});
        assertEquals(0.0f, result[0], 1e-5f);
    }

    @Test
    void logSoftmax_valuesAreNegativeOrZero() {
        float[] logits = {1.0f, 2.0f, 3.0f};
        float[] result = MathOps.logSoftmax(logits);

        for (float v : result) {
            assertTrue(v <= 0f, "logSoftmax value should be <= 0, got " + v);
        }
    }

    @Test
    void logSoftmax_expSumsToOne() {
        float[] logits = {1.0f, 2.0f, 3.0f};
        float[] result = MathOps.logSoftmax(logits);

        float sum = 0f;
        for (float v : result) {
            sum += (float) Math.exp(v);
        }
        assertEquals(1.0f, sum, 1e-5f);
    }

    @Test
    void logSoftmax_consistentWithSoftmax() {
        float[] logits = {1.0f, 5.0f, 2.0f};
        float[] softmax = MathOps.softmax(logits);
        float[] logSoftmax = MathOps.logSoftmax(logits);

        for (int i = 0; i < logits.length; i++) {
            assertEquals(Math.log(softmax[i]), logSoftmax[i], 1e-5f);
        }
    }

    @Test
    void logSoftmax_numericallyStableWithLargeValues() {
        float[] logits = {1000f, 1001f, 1002f};
        float[] result = MathOps.logSoftmax(logits);

        for (float v : result) {
            assertFalse(Float.isNaN(v), "logSoftmax produced NaN");
            assertFalse(Float.isInfinite(v), "logSoftmax produced Inf");
        }

        float sum = 0f;
        for (float v : result) {
            sum += (float) Math.exp(v);
        }
        assertEquals(1.0f, sum, 1e-3f);
    }

    @Test
    void topK_returnsIndicesOfLargestValues() {
        float[] values = {0.1f, 0.9f, 0.3f, 0.7f, 0.5f};
        int[] top3 = MathOps.topK(values, 3);

        assertEquals(3, top3.length);
        assertEquals(1, top3[0]); // 0.9
        assertEquals(3, top3[1]); // 0.7
        assertEquals(4, top3[2]); // 0.5
    }

    @Test
    void topK_sortedDescending() {
        float[] values = {0.2f, 0.8f, 0.6f, 0.4f};
        int[] top = MathOps.topK(values, 4);

        for (int i = 1; i < top.length; i++) {
            assertTrue(values[top[i - 1]] >= values[top[i]],
                    "topK result not sorted descending at index " + i);
        }
    }

    @Test
    void topK_kLargerThanArray_returnsAll() {
        float[] values = {0.3f, 0.1f};
        int[] top = MathOps.topK(values, 10);

        assertEquals(2, top.length);
        assertEquals(0, top[0]); // 0.3
        assertEquals(1, top[1]); // 0.1
    }

    @Test
    void topK_singleElement() {
        float[] values = {0.5f};
        int[] top = MathOps.topK(values, 1);

        assertEquals(1, top.length);
        assertEquals(0, top[0]);
    }

    @Test
    void topK_kEqualsZero_returnsEmpty() {
        float[] values = {0.1f, 0.2f};
        int[] top = MathOps.topK(values, 0);

        assertEquals(0, top.length);
    }
}
