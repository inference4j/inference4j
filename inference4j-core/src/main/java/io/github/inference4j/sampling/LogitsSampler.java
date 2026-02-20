package io.github.inference4j.sampling;

@FunctionalInterface
public interface LogitsSampler {
    int sample(float[] logits);
}
