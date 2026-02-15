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

/**
 * Numerical utilities for post-processing model outputs.
 */
public final class MathOps {

    private MathOps() {
    }

    /**
     * Computes softmax over the input array (numerically stable).
     * Subtracts the max value before exponentiation to avoid overflow.
     */
    public static float[] softmax(float[] logits) {
        float[] result = new float[logits.length];

        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            if (v > max) {
                max = v;
            }
        }

        float sum = 0f;
        for (int i = 0; i < logits.length; i++) {
            result[i] = (float) Math.exp(logits[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

    /**
     * Computes element-wise sigmoid over the input array: {@code 1 / (1 + exp(-x))}.
     */
    public static float[] sigmoid(float[] values) {
        float[] result = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = (float) (1.0 / (1.0 + Math.exp(-values[i])));
        }
        return result;
    }

    /**
     * Computes log-softmax over the input array (numerically stable).
     * Computed as {@code x - log(sum(exp(x)))} using the max-subtraction trick.
     */
    public static float[] logSoftmax(float[] logits) {
        float[] result = new float[logits.length];

        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            if (v > max) {
                max = v;
            }
        }

        float sum = 0f;
        for (float logit : logits) {
            sum += (float) Math.exp(logit - max);
        }
        float logSumExp = max + (float) Math.log(sum);

        for (int i = 0; i < logits.length; i++) {
            result[i] = logits[i] - logSumExp;
        }

        return result;
    }

    /**
     * Applies non-maximum suppression (NMS) to a set of bounding boxes.
     *
     * <p>Greedily selects high-confidence boxes and removes overlapping boxes
     * whose IoU (Intersection over Union) exceeds the threshold. Boxes are
     * represented as {@code [x1, y1, x2, y2]} (top-left and bottom-right corners).
     *
     * @param boxes  array of length {@code N*4}, where each group of 4 floats is
     *               {@code [x1, y1, x2, y2]}
     * @param scores confidence score for each box (length {@code N})
     * @param iouThreshold boxes with IoU above this value are suppressed
     * @return indices of the kept boxes, sorted by descending score
     */
    public static int[] nms(float[] boxes, float[] scores, float iouThreshold) {
        int n = scores.length;
        if (n == 0) {
            return new int[0];
        }

        // Sort indices by score descending
        int[] order = topK(scores, n);

        boolean[] suppressed = new boolean[n];
        int kept = 0;
        int[] keepBuffer = new int[n];

        for (int i = 0; i < n; i++) {
            int idx = order[i];
            if (suppressed[idx]) {
                continue;
            }
            keepBuffer[kept++] = idx;

            float x1i = boxes[idx * 4];
            float y1i = boxes[idx * 4 + 1];
            float x2i = boxes[idx * 4 + 2];
            float y2i = boxes[idx * 4 + 3];
            float areaI = (x2i - x1i) * (y2i - y1i);

            for (int j = i + 1; j < n; j++) {
                int jdx = order[j];
                if (suppressed[jdx]) {
                    continue;
                }

                float x1j = boxes[jdx * 4];
                float y1j = boxes[jdx * 4 + 1];
                float x2j = boxes[jdx * 4 + 2];
                float y2j = boxes[jdx * 4 + 3];

                float interX1 = Math.max(x1i, x1j);
                float interY1 = Math.max(y1i, y1j);
                float interX2 = Math.min(x2i, x2j);
                float interY2 = Math.min(y2i, y2j);

                float interArea = Math.max(0f, interX2 - interX1) * Math.max(0f, interY2 - interY1);
                float areaJ = (x2j - x1j) * (y2j - y1j);
                float iou = interArea / (areaI + areaJ - interArea);

                if (iou > iouThreshold) {
                    suppressed[jdx] = true;
                }
            }
        }

        int[] result = new int[kept];
        System.arraycopy(keepBuffer, 0, result, 0, kept);
        return result;
    }

    /**
     * Converts bounding boxes from center format {@code [cx, cy, w, h]} to corner
     * format {@code [x1, y1, x2, y2]}.
     *
     * <p>The input array is a flat sequence of box coordinates:
     * {@code [cx0, cy0, w0, h0, cx1, cy1, w1, h1, ...]}. The returned array has
     * the same length with each group of four values converted.
     *
     * @param boxes flat array of length {@code N*4} in {@code [cx, cy, w, h]} format
     * @return flat array of length {@code N*4} in {@code [x1, y1, x2, y2]} format
     */
    public static float[] cxcywh2xyxy(float[] boxes) {
        float[] result = new float[boxes.length];
        for (int i = 0; i < boxes.length; i += 4) {
            float cx = boxes[i];
            float cy = boxes[i + 1];
            float halfW = boxes[i + 2] / 2f;
            float halfH = boxes[i + 3] / 2f;
            result[i]     = cx - halfW;  // x1
            result[i + 1] = cy - halfH;  // y1
            result[i + 2] = cx + halfW;  // x2
            result[i + 3] = cy + halfH;  // y2
        }
        return result;
    }

    /**
     * Performs CTC greedy decoding on logits.
     *
     * <p>For each timestep, takes the argmax over the vocabulary dimension,
     * then collapses consecutive repeated tokens and removes blank tokens.
     *
     * @param logits     flat logit array of shape {@code [timeSteps, vocabSize]}
     * @param timeSteps  number of timesteps
     * @param vocabSize  vocabulary size
     * @param blankIndex index of the CTC blank token
     * @return decoded token indices (repeats collapsed, blanks removed)
     */
    public static int[] ctcGreedyDecode(float[] logits, int timeSteps, int vocabSize, int blankIndex) {
        // Step 1: argmax per timestep
        int[] argmax = new int[timeSteps];
        for (int t = 0; t < timeSteps; t++) {
            int bestIdx = 0;
            float bestVal = logits[t * vocabSize];
            for (int v = 1; v < vocabSize; v++) {
                float val = logits[t * vocabSize + v];
                if (val > bestVal) {
                    bestVal = val;
                    bestIdx = v;
                }
            }
            argmax[t] = bestIdx;
        }

        // Step 2: collapse consecutive repeats and remove blanks
        int[] buffer = new int[timeSteps];
        int count = 0;
        int prev = -1;
        for (int t = 0; t < timeSteps; t++) {
            int token = argmax[t];
            if (token != prev) {
                if (token != blankIndex) {
                    buffer[count++] = token;
                }
                prev = token;
            }
        }

        int[] result = new int[count];
        System.arraycopy(buffer, 0, result, 0, count);
        return result;
    }

    /**
     * Returns the indices of the top-K largest values, sorted descending by value.
     * Uses partial selection sort — O(n*k), efficient for small k on large arrays.
     */
    public static int[] topK(float[] values, int k) {
        k = Math.min(k, values.length);
        int[] indices = new int[values.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }

        for (int i = 0; i < k; i++) {
            int maxIdx = i;
            for (int j = i + 1; j < indices.length; j++) {
                if (values[indices[j]] > values[indices[maxIdx]]) {
                    maxIdx = j;
                }
            }
            int tmp = indices[i];
            indices[i] = indices[maxIdx];
            indices[maxIdx] = tmp;
        }

        int[] result = new int[k];
        System.arraycopy(indices, 0, result, 0, k);
        return result;
    }
}
