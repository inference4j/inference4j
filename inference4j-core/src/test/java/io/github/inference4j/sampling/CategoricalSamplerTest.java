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

import java.util.HashSet;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;

class CategoricalSamplerTest {

    private final CategoricalSampler sampler = new CategoricalSampler();

    @Test
    void sample_concentratedDistribution_picksDominantToken() {
        // Softmax of [100, -100, -100] is essentially [1.0, 0.0, 0.0]
        float[] logits = {100.0f, -100.0f, -100.0f};

        for (int i = 0; i < 100; i++) {
            int result = sampler.sample(logits);
            assertThat(result).isEqualTo(0);
        }
    }

    @Test
    void sample_returnsValidIndex() {
        float[] logits = {1.0f, 2.0f, 3.0f, 4.0f};

        for (int i = 0; i < 100; i++) {
            int result = sampler.sample(logits);
            assertThat(result).isBetween(0, logits.length - 1);
        }
    }

    @Test
    void sample_distributionCheck() {
        // Uniform logits should produce all indices over many runs
        float[] logits = {0.0f, 0.0f, 0.0f};
        Set<Integer> observed = new HashSet<>();

        for (int i = 0; i < 1000; i++) {
            observed.add(sampler.sample(logits));
            if (observed.size() == logits.length) break;
        }

        assertThat(observed).containsExactlyInAnyOrder(0, 1, 2);
    }
}
