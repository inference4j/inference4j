package io.github.inference4j.metrics;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MicrometerRouterMetricsTest {

    private SimpleMeterRegistry registry;
    private MicrometerRouterMetrics metrics;

    @BeforeEach
    void setUp() {
        registry = new SimpleMeterRegistry();
        metrics = new MicrometerRouterMetrics(registry);
    }

    @Test
    void recordsSuccessTimer() {
        metrics.recordSuccess("my-router", "route-a", 1_000_000);

        Timer timer = registry.find("inference4j.router.duration")
                .tag("router", "my-router")
                .tag("route", "route-a")
                .tag("outcome", "success")
                .timer();

        assertNotNull(timer);
        assertEquals(1, timer.count());
    }

    @Test
    void recordsFailureTimer() {
        metrics.recordFailure("my-router", "route-a", 2_000_000);

        Timer timer = registry.find("inference4j.router.duration")
                .tag("router", "my-router")
                .tag("route", "route-a")
                .tag("outcome", "failure")
                .timer();

        assertNotNull(timer);
        assertEquals(1, timer.count());
    }

    @Test
    void recordsShadowTimerAndCounter() {
        metrics.recordShadow("my-router", "shadow-route", 500_000, true);

        Timer timer = registry.find("inference4j.router.duration")
                .tag("router", "my-router")
                .tag("route", "shadow-route")
                .tag("outcome", "shadow")
                .timer();

        assertNotNull(timer);
        assertEquals(1, timer.count());

        Counter counter = registry.find("inference4j.router.shadow")
                .tag("router", "my-router")
                .tag("route", "shadow-route")
                .tag("success", "true")
                .counter();

        assertNotNull(counter);
        assertEquals(1.0, counter.count());
    }

    @Test
    void recordsShadowFailure() {
        metrics.recordShadow("my-router", "shadow-route", 500_000, false);

        Counter counter = registry.find("inference4j.router.shadow")
                .tag("router", "my-router")
                .tag("route", "shadow-route")
                .tag("success", "false")
                .counter();

        assertNotNull(counter);
        assertEquals(1.0, counter.count());
    }

    @Test
    void separatesMetricsByRoute() {
        metrics.recordSuccess("router", "route-a", 1_000_000);
        metrics.recordSuccess("router", "route-b", 1_000_000);
        metrics.recordSuccess("router", "route-a", 1_000_000);

        Timer timerA = registry.find("inference4j.router.duration")
                .tag("route", "route-a")
                .tag("outcome", "success")
                .timer();

        Timer timerB = registry.find("inference4j.router.duration")
                .tag("route", "route-b")
                .tag("outcome", "success")
                .timer();

        assertNotNull(timerA);
        assertNotNull(timerB);
        assertEquals(2, timerA.count());
        assertEquals(1, timerB.count());
    }
}
