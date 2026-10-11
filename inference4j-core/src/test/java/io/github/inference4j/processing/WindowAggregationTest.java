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

package io.github.inference4j.processing;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class WindowAggregationTest {

    private static final List<float[]> WINDOWS = List.of(
            new float[]{0.9f, 0.1f},
            new float[]{0.2f, 0.8f},
            new float[]{0.7f, 0.3f});

    @Test
    void meanAveragesEachLabel() {
        float[] result = WindowAggregation.mean().aggregate(WINDOWS);

        assertThat(result[0]).isCloseTo(0.6f, within(1e-6f));
        assertThat(result[1]).isCloseTo(0.4f, within(1e-6f));
    }

    @Test
    void meanOfDistributionsIsADistribution() {
        float[] result = WindowAggregation.mean().aggregate(WINDOWS);

        assertThat(result[0] + result[1]).isCloseTo(1f, within(1e-6f));
    }

    @Test
    void maxKeepsTheHighestScoreOfEachLabel() {
        assertThat(WindowAggregation.max().aggregate(WINDOWS)).containsExactly(0.9f, 0.8f);
    }

    @Test
    void maxDoesNotModifyTheWindowScores() {
        float[] first = {0.1f, 0.2f};

        WindowAggregation.max().aggregate(List.of(first, new float[]{0.5f, 0.6f}));

        assertThat(first).containsExactly(0.1f, 0.2f);
    }

    @Test
    void aSingleWindowIsReturnedAsIs() {
        List<float[]> one = List.of(new float[]{0.3f, 0.7f});

        assertThat(WindowAggregation.mean().aggregate(one)).containsExactly(0.3f, 0.7f);
        assertThat(WindowAggregation.max().aggregate(one)).containsExactly(0.3f, 0.7f);
    }
}
