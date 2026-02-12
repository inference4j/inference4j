package io.github.inference4j.metrics;

/**
 * No-op implementation used when metrics are not configured.
 */
public final class NoOpRouterMetrics implements RouterMetrics {

    private static final NoOpRouterMetrics INSTANCE = new NoOpRouterMetrics();

    private NoOpRouterMetrics() {}

    public static NoOpRouterMetrics getInstance() {
        return INSTANCE;
    }

    @Override
    public void recordSuccess(String routerName, String routeName, long durationNanos) {}

    @Override
    public void recordFailure(String routerName, String routeName, long durationNanos) {}

    @Override
    public void recordShadow(String routerName, String routeName, long durationNanos, boolean success) {}
}
