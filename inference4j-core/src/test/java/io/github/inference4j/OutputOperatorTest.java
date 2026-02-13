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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class OutputOperatorTest {

    @Test
    void softmax_delegatesToMathOps() {
        float[] logits = {1.0f, 2.0f, 3.0f};
        float[] result = OutputOperator.softmax().apply(logits);

        assertArrayEquals(MathOps.softmax(logits), result, 1e-6f);
    }

    @Test
    void sigmoid_delegatesToMathOps() {
        float[] values = {-1.0f, 0.0f, 1.0f};
        float[] result = OutputOperator.sigmoid().apply(values);

        assertArrayEquals(MathOps.sigmoid(values), result, 1e-6f);
    }

    @Test
    void logSoftmax_delegatesToMathOps() {
        float[] logits = {1.0f, 2.0f, 3.0f};
        float[] result = OutputOperator.logSoftmax().apply(logits);

        assertArrayEquals(MathOps.logSoftmax(logits), result, 1e-6f);
    }

    @Test
    void identity_returnsInputUnchanged() {
        float[] values = {0.1f, 0.5f, 0.4f};
        float[] result = OutputOperator.identity().apply(values);

        assertSame(values, result);
    }

    @Test
    void andThen_composesOperators() {
        // Scale all values by 2, then apply softmax
        OutputOperator doubler = values -> {
            float[] scaled = new float[values.length];
            for (int i = 0; i < values.length; i++) {
                scaled[i] = values[i] * 2;
            }
            return scaled;
        };

        OutputOperator composed = doubler.andThen(OutputOperator.softmax());
        float[] logits = {1.0f, 2.0f, 3.0f};
        float[] result = composed.apply(logits);

        // Should equal softmax of doubled values
        float[] expected = MathOps.softmax(new float[]{2.0f, 4.0f, 6.0f});
        assertArrayEquals(expected, result, 1e-6f);
    }

    @Test
    void andThen_rejectsNull() {
        assertThrows(NullPointerException.class, () -> OutputOperator.identity().andThen(null));
    }

    @Test
    void lambda_worksAsOutputOperator() {
        OutputOperator clamp = values -> {
            float[] result = new float[values.length];
            for (int i = 0; i < values.length; i++) {
                result[i] = Math.max(0f, Math.min(1f, values[i]));
            }
            return result;
        };

        float[] result = clamp.apply(new float[]{-0.5f, 0.5f, 1.5f});

        assertArrayEquals(new float[]{0f, 0.5f, 1f}, result, 1e-6f);
    }
}
