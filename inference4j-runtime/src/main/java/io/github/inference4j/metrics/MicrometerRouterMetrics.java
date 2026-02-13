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
