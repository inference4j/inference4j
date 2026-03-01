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

class LogitsProcessorsTest {

    @Test
    void temperature_returnsTemperatureProcessor() {
        LogitsProcessor processor = LogitsProcessors.temperature(2.0f);
        float[] logits = {4.0f, 6.0f, 10.0f};

        float[] result = processor.process(logits);

        assertThat(result[0]).isCloseTo(2.0f, within(1e-6f));
        assertThat(result[1]).isCloseTo(3.0f, within(1e-6f));
        assertThat(result[2]).isCloseTo(5.0f, within(1e-6f));
    }

    @Test
    void topK_returnsTopKProcessor() {
        LogitsProcessor processor = LogitsProcessors.topK(2);
        float[] logits = {1.0f, 5.0f, 3.0f, 2.0f};

        float[] result = processor.process(logits);

        assertThat(result[1]).isEqualTo(5.0f);
        assertThat(result[2]).isEqualTo(3.0f);
        assertThat(result[0]).isEqualTo(Float.NEGATIVE_INFINITY);
        assertThat(result[3]).isEqualTo(Float.NEGATIVE_INFINITY);
    }

    @Test
    void topP_returnsTopPProcessor() {
        LogitsProcessor processor = LogitsProcessors.topP(0.01f);
        // Index 2 is dominant
        float[] logits = {-10.0f, -10.0f, 10.0f, -10.0f};

        float[] result = processor.process(logits);

        assertThat(result[2]).isEqualTo(10.0f);
        assertThat(result[0]).isEqualTo(Float.NEGATIVE_INFINITY);
        assertThat(result[1]).isEqualTo(Float.NEGATIVE_INFINITY);
        assertThat(result[3]).isEqualTo(Float.NEGATIVE_INFINITY);
    }
}
