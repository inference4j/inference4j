package io.github.inference4j.routing;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RoundRobinRoutingStrategyTest {

    @Test
    void cyclesThroughRoutesInOrder() {
        RoundRobinRoutingStrategy strategy = new RoundRobinRoutingStrategy();

        Route<String> a = Route.of("a", "model-a", 1);
        Route<String> b = Route.of("b", "model-b", 1);
        Route<String> c = Route.of("c", "model-c", 1);
        List<Route<String>> routes = List.of(a, b, c);

        assertEquals(a, strategy.select(routes));
        assertEquals(b, strategy.select(routes));
        assertEquals(c, strategy.select(routes));
        assertEquals(a, strategy.select(routes));
        assertEquals(b, strategy.select(routes));
    }

    @Test
    void worksWithSingleRoute() {
        RoundRobinRoutingStrategy strategy = new RoundRobinRoutingStrategy();

        Route<String> only = Route.of("only", "model", 1);
        List<Route<String>> routes = List.of(only);

        for (int i = 0; i < 10; i++) {
            assertEquals(only, strategy.select(routes));
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
            assertEquals(a, strategy.select(routes), "Cycle " + cycle + " first");
            assertEquals(b, strategy.select(routes), "Cycle " + cycle + " second");
        }
    }
}
