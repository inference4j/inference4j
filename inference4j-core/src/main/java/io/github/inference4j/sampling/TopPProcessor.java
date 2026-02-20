package io.github.inference4j.sampling;

import io.github.inference4j.processing.MathOps;

import java.util.Arrays;

public class TopPProcessor implements LogitsProcessor {

    private final float p;

    public TopPProcessor(float p) {
        this.p = p;
    }

    @Override
    public float[] process(float[] logits) {
        if (p >= 1.0f) return logits;

        float[] probs = MathOps.softmax(logits);

        int n = logits.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));

        float cumulative = 0;
        boolean[] keep = new boolean[n];
        for (int i = 0; i < n; i++) {
            cumulative += probs[indices[i]];
            keep[indices[i]] = true;
            if (cumulative >= p) break;
        }

        float[] result = logits.clone();
        for (int i = 0; i < n; i++) {
            if (!keep[i]) {
                result[i] = Float.NEGATIVE_INFINITY;
            }
        }
        return result;
    }
}
