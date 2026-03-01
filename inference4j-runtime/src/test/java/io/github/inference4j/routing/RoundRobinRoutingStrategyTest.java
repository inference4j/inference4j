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

package io.github.inference4j.routing;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class RoundRobinRoutingStrategyTest {

    @Test
    void cyclesThroughRoutesInOrder() {
        RoundRobinRoutingStrategy strategy = new RoundRobinRoutingStrategy();

        Route<String> a = Route.of("a", "model-a", 1);
        Route<String> b = Route.of("b", "model-b", 1);
        Route<String> c = Route.of("c", "model-c", 1);
        List<Route<String>> routes = List.of(a, b, c);

        assertThat(strategy.select(routes)).isEqualTo(a);
        assertThat(strategy.select(routes)).isEqualTo(b);
        assertThat(strategy.select(routes)).isEqualTo(c);
        assertThat(strategy.select(routes)).isEqualTo(a);
        assertThat(strategy.select(routes)).isEqualTo(b);
    }

    @Test
    void worksWithSingleRoute() {
        RoundRobinRoutingStrategy strategy = new RoundRobinRoutingStrategy();

        Route<String> only = Route.of("only", "model", 1);
        List<Route<String>> routes = List.of(only);

        for (int i = 0; i < 10; i++) {
            assertThat(strategy.select(routes)).isEqualTo(only);
        }
    }

    @Test
    void wrapsAroundCorrectly() {
        RoundRobinRoutingStrategy strategy = new RoundRobinRoutingStrategy();

        Route<String> a = Route.of("a", "model-a", 1);
        Route<String> b = Route.of("b", "model-b", 1);
        List<Route<String>> routes = List.of(a, b);

        // Run through multiple full cycles
        for (int cycle = 0; cycle < 5; cycle++) {
            assertThat(strategy.select(routes)).as("Cycle " + cycle + " first").isEqualTo(a);
            assertThat(strategy.select(routes)).as("Cycle " + cycle + " second").isEqualTo(b);
        }
    }
}
