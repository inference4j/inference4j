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

class GreedySamplerTest {

    private final GreedySampler sampler = new GreedySampler();

    @Test
    void sample_returnsArgmaxIndex() {
        float[] logits = {1.0f, 3.0f, 2.0f, 0.5f};

        int result = sampler.sample(logits);

        assertThat(result).isEqualTo(1);
    }

    @Test
    void sample_tieBreaking_returnsFirstOccurrence() {
        float[] logits = {1.0f, 5.0f, 5.0f, 3.0f};

        int result = sampler.sample(logits);

        // Implementation uses strict '>' so the first max index (1) is returned
        assertThat(result).isEqualTo(1);
    }

    @Test
    void sample_singleElement_returnsThatIndex() {
        float[] logits = {42.0f};

        int result = sampler.sample(logits);

        assertThat(result).isEqualTo(0);
    }

    @Test
    void sample_allNegativeValues_returnsLargest() {
        float[] logits = {-5.0f, -1.0f, -3.0f, -10.0f};

        int result = sampler.sample(logits);

        assertThat(result).isEqualTo(1); // -1.0 is the largest
    }
}
