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
