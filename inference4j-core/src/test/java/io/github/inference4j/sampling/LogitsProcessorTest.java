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

package io.github.inference4j.sampling;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class LogitsProcessorTest {

    @Test
    void identity_returnsInputUnchanged() {
        LogitsProcessor identity = LogitsProcessor.identity();
        float[] logits = {1.0f, 2.0f, 3.0f};

        float[] result = identity.process(logits);

        assertThat(result).isSameAs(logits);
    }

    @Test
    void andThen_composesInOrder() {
        // First processor: adds 1 to each element
        LogitsProcessor addOne = logits -> {
            float[] result = logits.clone();
            for (int i = 0; i < result.length; i++) result[i] += 1;
            return result;
        };

        // Second processor: multiplies each element by 2
        LogitsProcessor timesTwo = logits -> {
            float[] result = logits.clone();
            for (int i = 0; i < result.length; i++) result[i] *= 2;
            return result;
        };

        LogitsProcessor composed = addOne.andThen(timesTwo);
        float[] logits = {1.0f, 2.0f, 3.0f};

        float[] result = composed.process(logits);

        // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
        assertThat(result[0]).isCloseTo(4.0f, within(1e-6f));
        assertThat(result[1]).isCloseTo(6.0f, within(1e-6f));
        assertThat(result[2]).isCloseTo(8.0f, within(1e-6f));
    }

    @Test
    void andThen_threeWayChain() {
        LogitsProcessor addOne = logits -> {
            float[] result = logits.clone();
            for (int i = 0; i < result.length; i++) result[i] += 1;
            return result;
        };

        LogitsProcessor timesTwo = logits -> {
            float[] result = logits.clone();
            for (int i = 0; i < result.length; i++) result[i] *= 2;
            return result;
        };

        LogitsProcessor subtractThree = logits -> {
            float[] result = logits.clone();
            for (int i = 0; i < result.length; i++) result[i] -= 3;
            return result;
        };

        LogitsProcessor composed = addOne.andThen(timesTwo).andThen(subtractThree);
        float[] logits = {5.0f};

        float[] result = composed.process(logits);

        // (5+1)*2 - 3 = 9
        assertThat(result[0]).isCloseTo(9.0f, within(1e-6f));
    }
}
