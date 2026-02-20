package io.github.inference4j.sampling;

import io.github.inference4j.processing.MathOps;

import java.util.concurrent.ThreadLocalRandom;

public class CategoricalSampler implements LogitsSampler {
    @Override
    public int sample(float[] logits) {
        var probs = MathOps.softmax(logits);
        var random = ThreadLocalRandom.current().nextFloat();
        var sum = 0f;
        for (int i = 0; i < probs.length; i++) {
            sum += probs[i];
            if (sum >= random) {
                return i;
            }
        }
        return probs.length - 1;
    }
}
