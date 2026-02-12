package io.github.inference4j.routing;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Selects a route randomly, proportional to route weights.
 * Thread-safe via {@link ThreadLocalRandom}.
 */
public final class WeightedRoutingStrategy implements RoutingStrategy {

    @Override
    public <T> Route<T> select(List<Route<T>> routes) {
        int totalWeight = 0;
        for (Route<T> route : routes) {
            totalWeight += route.weight();
        }

        int random = ThreadLocalRandom.current().nextInt(totalWeight);
        int cumulative = 0;

        for (Route<T> route : routes) {
            cumulative += route.weight();
            if (random < cumulative) {
                return route;
            }
        }

        // Should never happen if weights are positive
        return routes.getLast();
    }
}
