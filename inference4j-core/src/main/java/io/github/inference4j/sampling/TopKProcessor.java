package io.github.inference4j.sampling;

import java.util.Arrays;

public class TopKProcessor implements LogitsProcessor {

    private final int k;

    public TopKProcessor(int k) {
        this.k = k;
    }

    @Override
    public float[] process(float[] logits) {
        if (k <= 0 || k >= logits.length) return logits;

        float[] sorted = logits.clone();
        Arrays.sort(sorted);
        float threshold = sorted[sorted.length - k];

        float[] result = logits.clone();
        for (int i = 0; i < result.length; i++) {
            if (result[i] < threshold) {
                result[i] = Float.NEGATIVE_INFINITY;
            }
        }
        return result;
    }
}
