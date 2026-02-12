package io.github.inference4j.routing;

import java.util.List;

/**
 * Selects a route from available candidates.
 */
@FunctionalInterface
public interface RoutingStrategy {

    <T> Route<T> select(List<Route<T>> routes);
}
