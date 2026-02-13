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

import java.util.Objects;

/**
 * A named model route with a weight for traffic splitting.
 * Shadow routes run after the primary route but their results are discarded.
 */
public final class Route<T> {

    private final String name;
    private final T model;
    private final int weight;
    private final boolean shadow;

    private Route(String name, T model, int weight, boolean shadow) {
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(model, "model must not be null");
        if (weight < 0) {
            throw new IllegalArgumentException("weight must be non-negative");
        }
        this.name = name;
        this.model = model;
        this.weight = weight;
        this.shadow = shadow;
    }

    public static <T> Route<T> of(String name, T model, int weight) {
        if (weight <= 0) {
            throw new IllegalArgumentException("weight must be positive");
        }
        return new Route<>(name, model, weight, false);
    }

    public static <T> Route<T> shadow(String name, T model) {
        return new Route<>(name, model, 0, true);
    }

    public String name() {
        return name;
    }

    public T model() {
        return model;
    }

    public int weight() {
        return weight;
    }

    public boolean isShadow() {
        return shadow;
    }

    @Override
    public String toString() {
        return "Route{name='" + name + "', weight=" + weight + ", shadow=" + shadow + "}";
    }
}
