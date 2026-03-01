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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class WeightedRoutingStrategyTest {

    private final WeightedRoutingStrategy strategy = new WeightedRoutingStrategy();

    @Test
    void selectsSingleRoute() {
        Route<String> route = Route.of("only", "model-a", 1);
        List<Route<String>> routes = List.of(route);

        for (int i = 0; i < 100; i++) {
            assertThat(strategy.select(routes)).isEqualTo(route);
        }
    }

    @Test
    void distributesAccordingToWeights() {
        Route<String> heavy = Route.of("heavy", "model-a", 80);
        Route<String> light = Route.of("light", "model-b", 20);
        List<Route<String>> routes = List.of(heavy, light);

        Map<String, Integer> counts = new HashMap<>();
        int iterations = 10_000;

        for (int i = 0; i < iterations; i++) {
            Route<String> selected = strategy.select(routes);
            counts.merge(selected.name(), 1, Integer::sum);
        }

        double heavyRatio = counts.getOrDefault("heavy", 0) / (double) iterations;
        // Should be roughly 0.8 — allow wide tolerance for randomness
        assertThat(heavyRatio).as("Expected ~80%% heavy, got " + (heavyRatio * 100) + "%%")
                .isBetween(0.7, 0.9);
    }

    @Test
    void handlesEqualWeights() {
        Route<String> a = Route.of("a", "model-a", 1);
        Route<String> b = Route.of("b", "model-b", 1);
        Route<String> c = Route.of("c", "model-c", 1);
        List<Route<String>> routes = List.of(a, b, c);

        Map<String, Integer> counts = new HashMap<>();
        int iterations = 9_000;

        for (int i = 0; i < iterations; i++) {
            Route<String> selected = strategy.select(routes);
            counts.merge(selected.name(), 1, Integer::sum);
        }

        // Each should get roughly 1/3
        for (String name : List.of("a", "b", "c")) {
            double ratio = counts.getOrDefault(name, 0) / (double) iterations;
            assertThat(ratio).as("Expected ~33%% for " + name + ", got " + (ratio * 100) + "%%")
                    .isBetween(0.25, 0.42);
        }
    }
}
