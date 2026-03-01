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

import io.github.inference4j.processing.MathOps;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class TemperatureProcessorTest {

    @Test
    void process_dividesLogitsByTemperature() {
        var processor = new TemperatureProcessor(2.0f);
        float[] logits = {4.0f, 6.0f, 10.0f};

        float[] result = processor.process(logits);

        assertThat(result[0]).isCloseTo(2.0f, within(1e-6f));
        assertThat(result[1]).isCloseTo(3.0f, within(1e-6f));
        assertThat(result[2]).isCloseTo(5.0f, within(1e-6f));
    }

    @Test
    void process_temperatureOne_returnsIdenticalValues() {
        var processor = new TemperatureProcessor(1.0f);
        float[] logits = {1.0f, 2.0f, 3.0f};

        float[] result = processor.process(logits);

        assertThat(result).containsExactly(1.0f, 2.0f, 3.0f);
    }

    @Test
    void process_highTemperature_flattensProbabilities() {
        float[] logits = {1.0f, 5.0f, 2.0f};

        float[] baseProbs = MathOps.softmax(logits);
        float[] highTempLogits = new TemperatureProcessor(10.0f).process(logits);
        float[] highTempProbs = MathOps.softmax(highTempLogits);

        // High temperature makes distribution more uniform: max prob gets closer to 1/3
        float baseMax = Math.max(baseProbs[0], Math.max(baseProbs[1], baseProbs[2]));
        float highTempMax = Math.max(highTempProbs[0], Math.max(highTempProbs[1], highTempProbs[2]));
        assertThat(highTempMax).isLessThan(baseMax);

        // All probabilities should be closer to uniform (1/3)
        float uniform = 1.0f / 3;
        for (float prob : highTempProbs) {
            assertThat(Math.abs(prob - uniform)).isLessThan(Math.abs(baseProbs[0] - uniform) + 0.01f);
        }
    }

    @Test
    void process_lowTemperature_sharpensProbabilities() {
        float[] logits = {1.0f, 5.0f, 2.0f};

        float[] baseProbs = MathOps.softmax(logits);
        float[] lowTempLogits = new TemperatureProcessor(0.1f).process(logits);
        float[] lowTempProbs = MathOps.softmax(lowTempLogits);

        // Low temperature sharpens: the max probability increases
        float baseMax = Math.max(baseProbs[0], Math.max(baseProbs[1], baseProbs[2]));
        float lowTempMax = Math.max(lowTempProbs[0], Math.max(lowTempProbs[1], lowTempProbs[2]));
        assertThat(lowTempMax).isGreaterThan(baseMax);
    }

    @Test
    void process_doesNotMutateInput() {
        var processor = new TemperatureProcessor(2.0f);
        float[] logits = {4.0f, 6.0f, 10.0f};
        float[] originalCopy = logits.clone();

        processor.process(logits);

        assertThat(logits).containsExactly(originalCopy);
    }
}
