package io.github.inference4j.metrics;

/**
 * Records metrics for model routing operations.
 */
public interface RouterMetrics {

    void recordSuccess(String routerName, String routeName, long durationNanos);

    void recordFailure(String routerName, String routeName, long durationNanos);

    void recordShadow(String routerName, String routeName, long durationNanos, boolean success);
}
