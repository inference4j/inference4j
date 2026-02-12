package io.github.inference4j.metrics;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;

import java.time.Duration;
import java.util.Objects;

/**
 * Records routing metrics using Micrometer.
 * <p>
 * Timers: {@code inference4j.router.duration} with tags {@code router}, {@code route}, {@code outcome}.
 * Counters: {@code inference4j.router.shadow} with tags {@code router}, {@code route}, {@code success}.
 */
public final class MicrometerRouterMetrics implements RouterMetrics {

    private final MeterRegistry registry;

    public MicrometerRouterMetrics(MeterRegistry registry) {
        this.registry = Objects.requireNonNull(registry, "registry must not be null");
    }

    @Override
    public void recordSuccess(String routerName, String routeName, long durationNanos) {
        Timer.builder("inference4j.router.duration")
                .tag("router", routerName)
                .tag("route", routeName)
                .tag("outcome", "success")
                .register(registry)
                .record(Duration.ofNanos(durationNanos));
    }

    @Override
    public void recordFailure(String routerName, String routeName, long durationNanos) {
        Timer.builder("inference4j.router.duration")
                .tag("router", routerName)
                .tag("route", routeName)
                .tag("outcome", "failure")
                .register(registry)
                .record(Duration.ofNanos(durationNanos));
    }

    @Override
    public void recordShadow(String routerName, String routeName, long durationNanos, boolean success) {
        Timer.builder("inference4j.router.duration")
                .tag("router", routerName)
                .tag("route", routeName)
                .tag("outcome", "shadow")
                .register(registry)
                .record(Duration.ofNanos(durationNanos));

        Counter.builder("inference4j.router.shadow")
                .tag("router", routerName)
                .tag("route", routeName)
                .tag("success", String.valueOf(success))
                .register(registry)
                .increment();
    }
}
