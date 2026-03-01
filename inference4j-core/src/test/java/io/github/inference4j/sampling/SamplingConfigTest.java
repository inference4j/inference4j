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

class SamplingConfigTest {

    @Test
    void recordAccessors_returnCorrectValues() {
        var config = new SamplingConfig(0.7f, 50, 0.9f, 1.2f, 42L);

        assertThat(config.temperature()).isEqualTo(0.7f);
        assertThat(config.topK()).isEqualTo(50);
        assertThat(config.topP()).isEqualTo(0.9f);
        assertThat(config.repetitionPenalty()).isEqualTo(1.2f);
        assertThat(config.seed()).isEqualTo(42L);
    }

    @Test
    void equality_sameValues_areEqual() {
        var config1 = new SamplingConfig(0.7f, 50, 0.9f, 1.2f, 42L);
        var config2 = new SamplingConfig(0.7f, 50, 0.9f, 1.2f, 42L);

        assertThat(config1).isEqualTo(config2);
        assertThat(config1.hashCode()).isEqualTo(config2.hashCode());
    }

    @Test
    void equality_differentValues_areNotEqual() {
        var config1 = new SamplingConfig(0.7f, 50, 0.9f, 1.2f, 42L);
        var config2 = new SamplingConfig(1.0f, 40, 0.95f, 1.0f, 123L);

        assertThat(config1).isNotEqualTo(config2);
    }
}
