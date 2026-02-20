package io.github.inference4j.sampling;

public class GreedySampler implements LogitsSampler {
    @Override
    public int sample(float[] logits) {
        int maxIndex = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }
}
