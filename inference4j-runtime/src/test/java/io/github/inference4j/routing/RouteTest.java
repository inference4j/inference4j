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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class RouteTest {

    @Test
    void of_createsNonShadowRoute() {
        Object model = new Object();

        Route<Object> route = Route.of("v1", model, 5);

        assertThat(route.name()).isEqualTo("v1");
        assertThat(route.model()).isSameAs(model);
        assertThat(route.weight()).isEqualTo(5);
        assertThat(route.isShadow()).isFalse();
    }

    @Test
    void of_zeroWeight_throws() {
        assertThatThrownBy(() -> Route.of("v1", new Object(), 0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void of_negativeWeight_throws() {
        assertThatThrownBy(() -> Route.of("v1", new Object(), -1))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void of_nullName_throws() {
        assertThatThrownBy(() -> Route.of(null, new Object(), 1))
                .isInstanceOf(NullPointerException.class);
    }

    @Test
    void of_nullModel_throws() {
        assertThatThrownBy(() -> Route.of("v1", null, 1))
                .isInstanceOf(NullPointerException.class);
    }

    @Test
    void shadow_createsShadowRoute() {
        Object model = new Object();

        Route<Object> route = Route.shadow("shadow-v2", model);

        assertThat(route.name()).isEqualTo("shadow-v2");
        assertThat(route.model()).isSameAs(model);
        assertThat(route.weight()).isZero();
        assertThat(route.isShadow()).isTrue();
    }

    @Test
    void shadow_nullName_throws() {
        assertThatThrownBy(() -> Route.shadow(null, new Object()))
                .isInstanceOf(NullPointerException.class);
    }

    @Test
    void shadow_nullModel_throws() {
        assertThatThrownBy(() -> Route.shadow("s", null))
                .isInstanceOf(NullPointerException.class);
    }

    @Test
    void toString_containsNameAndWeight() {
        Route<Object> route = Route.of("v1", new Object(), 5);

        assertThat(route.toString()).contains("v1");
        assertThat(route.toString()).contains("5");
    }
}
