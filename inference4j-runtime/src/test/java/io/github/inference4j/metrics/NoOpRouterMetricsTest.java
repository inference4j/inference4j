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

package io.github.inference4j.metrics;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class NoOpRouterMetricsTest {

    @Test
    void getInstance_returnsSameInstance() {
        NoOpRouterMetrics first = NoOpRouterMetrics.getInstance();
        NoOpRouterMetrics second = NoOpRouterMetrics.getInstance();

        assertThat(first).isSameAs(second);
    }

    @Test
    void getInstance_isNotNull() {
        assertThat(NoOpRouterMetrics.getInstance()).isNotNull();
    }

    @Test
    void recordSuccess_doesNotThrow() {
        assertThatNoException().isThrownBy(() ->
                NoOpRouterMetrics.getInstance().recordSuccess("router", "route", 1_000_000));
    }

    @Test
    void recordFailure_doesNotThrow() {
        assertThatNoException().isThrownBy(() ->
                NoOpRouterMetrics.getInstance().recordFailure("router", "route", 2_000_000));
    }

    @Test
    void recordShadow_doesNotThrow() {
        assertThatNoException().isThrownBy(() ->
                NoOpRouterMetrics.getInstance().recordShadow("router", "route", 500_000, true));
    }
}
