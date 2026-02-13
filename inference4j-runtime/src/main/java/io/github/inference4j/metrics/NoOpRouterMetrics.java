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
