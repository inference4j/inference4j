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

package io.github.inference4j.processing;

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
    void nms_suppressesOverlappingBoxes() {
        // Two boxes with high overlap, different scores
        float[] boxes = {
                0f, 0f, 10f, 10f,  // box 0
                1f, 1f, 11f, 11f,  // box 1 (high overlap with box 0)
        };
        float[] scores = {0.9f, 0.8f};

        int[] kept = MathOps.nms(boxes, scores, 0.5f);

        assertEquals(1, kept.length);
        assertEquals(0, kept[0]); // higher score wins
    }

    @Test
    void nms_keepsNonOverlappingBoxes() {
        float[] boxes = {
                0f, 0f, 10f, 10f,    // box 0
                50f, 50f, 60f, 60f,  // box 1 (no overlap)
        };
        float[] scores = {0.9f, 0.8f};

        int[] kept = MathOps.nms(boxes, scores, 0.5f);

        assertEquals(2, kept.length);
        assertEquals(0, kept[0]);
        assertEquals(1, kept[1]);
    }

    @Test
    void nms_sortsByScoreDescending() {
        float[] boxes = {
                0f, 0f, 10f, 10f,
                50f, 50f, 60f, 60f,
                100f, 100f, 110f, 110f,
        };
        float[] scores = {0.5f, 0.9f, 0.7f};

        int[] kept = MathOps.nms(boxes, scores, 0.5f);

        assertEquals(3, kept.length);
        assertEquals(1, kept[0]); // 0.9
        assertEquals(2, kept[1]); // 0.7
        assertEquals(0, kept[2]); // 0.5
    }

    @Test
    void nms_emptyInput_returnsEmpty() {
        int[] kept = MathOps.nms(new float[0], new float[0], 0.5f);
        assertEquals(0, kept.length);
    }

    @Test
    void nms_singleBox_returnsThatBox() {
        float[] boxes = {0f, 0f, 10f, 10f};
        float[] scores = {0.95f};

        int[] kept = MathOps.nms(boxes, scores, 0.5f);

        assertEquals(1, kept.length);
        assertEquals(0, kept[0]);
    }

    @Test
    void nms_partialOverlapBelowThreshold_keepsBoth() {
        // Two boxes with partial overlap (~18% IoU)
        float[] boxes = {
                0f, 0f, 10f, 10f,   // area = 100
                7f, 7f, 17f, 17f,   // area = 100, intersection = 3*3 = 9, IoU = 9/191 ≈ 0.047
        };
        float[] scores = {0.9f, 0.8f};

        int[] kept = MathOps.nms(boxes, scores, 0.5f);

        assertEquals(2, kept.length);
    }

    @Test
    void nms_chainedSuppression_onlyKeepsHighest() {
        // Three nearly identical boxes — box 0 suppresses both 1 and 2
        float[] boxes = {
                0f, 0f, 10f, 10f,
                0f, 0f, 10f, 10f,
                0f, 0f, 10f, 10f,
        };
        float[] scores = {0.9f, 0.8f, 0.7f};

        int[] kept = MathOps.nms(boxes, scores, 0.5f);

        assertEquals(1, kept.length);
        assertEquals(0, kept[0]);
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

    @Test
    void ctcGreedyDecode_collapsesRepeatsAndRemovesBlanks() {
        // Vocab: 0=blank, 1=H, 2=E, 3=L, 4=O
        // Input sequence: H,H,_,E,E,L,L,L,O → H,E,L,O
        int vocabSize = 5;
        float[] logits = buildCtcLogits(vocabSize,
                1, 1, 0, 2, 2, 3, 3, 3, 4);

        int[] result = MathOps.ctcGreedyDecode(logits, 9, vocabSize, 0);

        assertArrayEquals(new int[]{1, 2, 3, 4}, result);
    }

    @Test
    void ctcGreedyDecode_allBlanks_returnsEmpty() {
        int vocabSize = 5;
        float[] logits = buildCtcLogits(vocabSize, 0, 0, 0);

        int[] result = MathOps.ctcGreedyDecode(logits, 3, vocabSize, 0);

        assertEquals(0, result.length);
    }

    @Test
    void ctcGreedyDecode_sameTokenSeparatedByBlank_keepsBoth() {
        // L, blank, L → L, L
        int vocabSize = 5;
        float[] logits = buildCtcLogits(vocabSize, 3, 0, 3);

        int[] result = MathOps.ctcGreedyDecode(logits, 3, vocabSize, 0);

        assertArrayEquals(new int[]{3, 3}, result);
    }

    @Test
    void ctcGreedyDecode_singleTimestep_blank() {
        int vocabSize = 3;
        float[] logits = buildCtcLogits(vocabSize, 0);

        int[] result = MathOps.ctcGreedyDecode(logits, 1, vocabSize, 0);

        assertEquals(0, result.length);
    }

    @Test
    void ctcGreedyDecode_singleTimestep_nonBlank() {
        int vocabSize = 3;
        float[] logits = buildCtcLogits(vocabSize, 2);

        int[] result = MathOps.ctcGreedyDecode(logits, 1, vocabSize, 0);

        assertArrayEquals(new int[]{2}, result);
    }

    /**
     * Builds synthetic CTC logits where the argmax at each timestep is the given token index.
     * Sets the target token to 10.0 and all others to -10.0.
     */
    private static float[] buildCtcLogits(int vocabSize, int... tokens) {
        float[] logits = new float[tokens.length * vocabSize];
        java.util.Arrays.fill(logits, -10.0f);
        for (int t = 0; t < tokens.length; t++) {
            logits[t * vocabSize + tokens[t]] = 10.0f;
        }
        return logits;
    }

    @Test
    void dotProduct_identicalNormalizedVectors_returnsOne() {
        float[] v = MathOps.l2Normalize(new float[]{3f, 4f});
        assertEquals(1.0f, MathOps.dotProduct(v, v), 1e-5f);
    }

    @Test
    void dotProduct_orthogonalVectors_returnsZero() {
        float[] a = {1f, 0f};
        float[] b = {0f, 1f};
        assertEquals(0f, MathOps.dotProduct(a, b), 1e-6f);
    }

    @Test
    void dotProduct_oppositeVectors_returnsNegativeOne() {
        float[] a = MathOps.l2Normalize(new float[]{1f, 0f});
        float[] b = MathOps.l2Normalize(new float[]{-1f, 0f});
        assertEquals(-1.0f, MathOps.dotProduct(a, b), 1e-5f);
    }

    @Test
    void dotProduct_mismatchedLengths_throws() {
        assertThrows(IllegalArgumentException.class, () ->
                MathOps.dotProduct(new float[]{1f, 2f}, new float[]{1f}));
    }

    @Test
    void cxcywh2xyxy_basicConversion() {
        // center=(50,50), size=20x30 → x1=40, y1=35, x2=60, y2=65
        float[] boxes = {50f, 50f, 20f, 30f};
        float[] result = MathOps.cxcywh2xyxy(boxes);

        assertEquals(40f, result[0], 1e-5f);
        assertEquals(35f, result[1], 1e-5f);
        assertEquals(60f, result[2], 1e-5f);
        assertEquals(65f, result[3], 1e-5f);
    }

    @Test
    void cxcywh2xyxy_multipleBoxes() {
        float[] boxes = {
                10f, 10f, 4f, 6f,   // → 8,7,12,13
                50f, 50f, 20f, 20f,  // → 40,40,60,60
        };
        float[] result = MathOps.cxcywh2xyxy(boxes);

        assertEquals(8, result.length);
        // box 0
        assertEquals(8f, result[0], 1e-5f);
        assertEquals(7f, result[1], 1e-5f);
        assertEquals(12f, result[2], 1e-5f);
        assertEquals(13f, result[3], 1e-5f);
        // box 1
        assertEquals(40f, result[4], 1e-5f);
        assertEquals(40f, result[5], 1e-5f);
        assertEquals(60f, result[6], 1e-5f);
        assertEquals(60f, result[7], 1e-5f);
    }

    @Test
    void cxcywh2xyxy_emptyInput() {
        float[] result = MathOps.cxcywh2xyxy(new float[0]);
        assertEquals(0, result.length);
    }

    @Test
    void cxcywh2xyxy_zeroSizeBox() {
        float[] boxes = {100f, 200f, 0f, 0f};
        float[] result = MathOps.cxcywh2xyxy(boxes);

        assertEquals(100f, result[0], 1e-5f);
        assertEquals(200f, result[1], 1e-5f);
        assertEquals(100f, result[2], 1e-5f);
        assertEquals(200f, result[3], 1e-5f);
    }
}
