package io.github.inference4j.sampling;

/**
 * Functional interface to apply transformation into logits before sampling.
 */
@FunctionalInterface
public interface LogitsProcessor {
    float[] process(float[] logits);

    default LogitsProcessor andThen(LogitsProcessor next) {
        return logits -> next.process(this.process(logits));
    }
}
