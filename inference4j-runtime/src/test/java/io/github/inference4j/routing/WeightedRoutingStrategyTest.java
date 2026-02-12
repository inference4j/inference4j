package io.github.inference4j.routing;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class WeightedRoutingStrategyTest {

    private final WeightedRoutingStrategy strategy = new WeightedRoutingStrategy();

    @Test
    void selectsSingleRoute() {
        Route<String> route = Route.of("only", "model-a", 1);
        List<Route<String>> routes = List.of(route);

        for (int i = 0; i < 100; i++) {
            assertEquals(route, strategy.select(routes));
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
        assertTrue(heavyRatio > 0.7 && heavyRatio < 0.9,
                "Expected ~80% heavy, got " + (heavyRatio * 100) + "%");
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
            assertTrue(ratio > 0.25 && ratio < 0.42,
                    "Expected ~33% for " + name + ", got " + (ratio * 100) + "%");
        }
    }
}
