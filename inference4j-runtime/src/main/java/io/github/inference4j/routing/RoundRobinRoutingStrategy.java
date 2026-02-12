package io.github.inference4j.routing;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Cycles through routes in order. Thread-safe via {@link AtomicLong}.
 */
public final class RoundRobinRoutingStrategy implements RoutingStrategy {

    private final AtomicLong counter = new AtomicLong();

    @Override
    public <T> Route<T> select(List<Route<T>> routes) {
        long index = counter.getAndIncrement();
        return routes.get((int) (index % routes.size()));
    }
}
