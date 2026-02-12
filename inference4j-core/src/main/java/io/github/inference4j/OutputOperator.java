package io.github.inference4j;

import java.util.Objects;

/**
 * A pluggable post-processing function for model outputs.
 *
 * <p>Transforms raw model output values (logits, probabilities, etc.) into the
 * desired output form. Common operations include softmax, sigmoid, and identity.
 *
 * <p>Built-in operators are available via static factory methods:
 * <pre>{@code
 * OutputOperator op = OutputOperator.softmax();
 * float[] probabilities = op.apply(logits);
 * }</pre>
 *
 * <p>Operators can be composed with {@link #andThen}:
 * <pre>{@code
 * OutputOperator scaled = myTemperatureScaling.andThen(OutputOperator.softmax());
 * }</pre>
 */
@FunctionalInterface
public interface OutputOperator {

    float[] apply(float[] values);

    static OutputOperator softmax() {
        return MathOps::softmax;
    }

    static OutputOperator sigmoid() {
        return MathOps::sigmoid;
    }

    static OutputOperator logSoftmax() {
        return MathOps::logSoftmax;
    }

    static OutputOperator identity() {
        return values -> values;
    }

    default OutputOperator andThen(OutputOperator after) {
        Objects.requireNonNull(after);
        return values -> after.apply(this.apply(values));
    }
}
