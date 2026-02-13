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
