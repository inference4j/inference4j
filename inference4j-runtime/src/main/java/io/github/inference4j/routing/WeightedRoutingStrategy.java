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
