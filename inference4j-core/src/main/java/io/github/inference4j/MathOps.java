package io.github.inference4j;

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
