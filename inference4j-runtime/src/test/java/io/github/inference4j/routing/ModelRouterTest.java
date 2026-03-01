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

import io.github.inference4j.metrics.RouterMetrics;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ModelRouterTest {

    interface TestModel extends AutoCloseable {
        String process(String input);
        @Override void close();
    }

    static class StubTestModel implements TestModel {
        private final String result;
        boolean closed = false;

        StubTestModel(String result) {
            this.result = result;
        }

        @Override
        public String process(String input) {
            return result + ":" + input;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    static class FailingTestModel implements TestModel {
        boolean closed = false;

        @Override
        public String process(String input) {
            throw new RuntimeException("model failure");
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    static class TestModelRouter extends ModelRouter<TestModel> {

        TestModelRouter(Builder builder) {
            super(builder);
        }

        String process(String input) {
            return execute(model -> model.process(input));
        }

        static Builder builder() {
            return new Builder();
        }

        static class Builder extends BaseBuilder<TestModel, Builder> {
            TestModelRouter build() {
                return new TestModelRouter(this);
            }
        }
    }

    static class RecordingMetrics implements RouterMetrics {
        final List<String> events = new ArrayList<>();

        @Override
        public void recordSuccess(String routerName, String routeName, long durationNanos) {
            events.add("success:" + routerName + ":" + routeName);
        }

        @Override
        public void recordFailure(String routerName, String routeName, long durationNanos) {
            events.add("failure:" + routerName + ":" + routeName);
        }

        @Override
        public void recordShadow(String routerName, String routeName, long durationNanos, boolean success) {
            events.add("shadow:" + routerName + ":" + routeName + ":" + success);
        }
    }

    @Test
    void routesToModel() {
        StubTestModel model = new StubTestModel("result");

        try (TestModelRouter router = TestModelRouter.builder()
                .name("test")
                .route("primary", model, 1)
                .build()) {

            assertThat(router.process("hello")).isEqualTo("result:hello");
        }
    }

    @Test
    void recordsSuccessMetrics() {
        RecordingMetrics metrics = new RecordingMetrics();

        try (TestModelRouter router = TestModelRouter.builder()
                .name("test")
                .route("primary", new StubTestModel("ok"), 1)
                .metrics(metrics)
                .build()) {

            router.process("hello");
        }

        assertThat(metrics.events).isEqualTo(List.of("success:test:primary"));
    }

    @Test
    void recordsFailureMetrics() {
        RecordingMetrics metrics = new RecordingMetrics();

        try (TestModelRouter router = TestModelRouter.builder()
                .name("test")
                .route("primary", new FailingTestModel(), 1)
                .metrics(metrics)
                .build()) {

            assertThatThrownBy(() -> router.process("hello")).isInstanceOf(RuntimeException.class);
        }

        assertThat(metrics.events).isEqualTo(List.of("failure:test:primary"));
    }

    @Test
    void executesShadowAfterPrimary() {
        RecordingMetrics metrics = new RecordingMetrics();
        StubTestModel primary = new StubTestModel("primary");
        StubTestModel shadow = new StubTestModel("shadow");

        try (TestModelRouter router = TestModelRouter.builder()
                .name("test")
                .route("primary", primary, 1)
                .shadow("shadow", shadow)
                .metrics(metrics)
                .build()) {

            String result = router.process("hello");
            assertThat(result).isEqualTo("primary:hello");
        }

        assertThat(metrics.events).hasSize(2);
        assertThat(metrics.events.get(0)).isEqualTo("success:test:primary");
        assertThat(metrics.events.get(1)).isEqualTo("shadow:test:shadow:true");
    }

    @Test
    void shadowFailureDoesNotPropagate() {
        StubTestModel primary = new StubTestModel("ok");
        FailingTestModel shadow = new FailingTestModel();

        try (TestModelRouter router = TestModelRouter.builder()
                .name("test")
                .route("primary", primary, 1)
                .shadow("shadow", shadow)
                .build()) {

            // Should not throw even though shadow fails
            String result = router.process("hello");
            assertThat(result).isEqualTo("ok:hello");
        }
    }

    @Test
    void closesModelsWhenOwned() {
        StubTestModel model1 = new StubTestModel("a");
        StubTestModel model2 = new StubTestModel("b");
        StubTestModel shadowModel = new StubTestModel("s");

        TestModelRouter router = TestModelRouter.builder()
                .name("test")
                .route("r1", model1, 1)
                .route("r2", model2, 1)
                .shadow("shadow", shadowModel)
                .build();

        router.close();

        assertThat(model1.closed).isTrue();
        assertThat(model2.closed).isTrue();
        assertThat(shadowModel.closed).isTrue();
    }

    @Test
    void doesNotCloseModelsWhenNotOwned() {
        StubTestModel model = new StubTestModel("a");

        TestModelRouter router = TestModelRouter.builder()
                .name("test")
                .route("r1", model, 1)
                .ownsModels(false)
                .build();

        router.close();

        assertThat(model.closed).isFalse();
    }

    @Test
    void throwsWhenNoRoutes() {
        assertThatThrownBy(() -> TestModelRouter.builder()
                        .name("test")
                        .build())
                .isInstanceOf(RoutingException.class);
    }

    @Test
    void throwsWhenNoName() {
        assertThatThrownBy(() -> TestModelRouter.builder()
                        .route("r", new StubTestModel("a"), 1)
                        .build())
                .isInstanceOf(NullPointerException.class);
    }

    @Test
    void usesRoundRobinStrategy() {
        StubTestModel modelA = new StubTestModel("a");
        StubTestModel modelB = new StubTestModel("b");

        try (TestModelRouter router = TestModelRouter.builder()
                .name("test")
                .route("r1", modelA, 1)
                .route("r2", modelB, 1)
                .strategy(new RoundRobinRoutingStrategy())
                .build()) {

            assertThat(router.process("x")).isEqualTo("a:x");
            assertThat(router.process("x")).isEqualTo("b:x");
            assertThat(router.process("x")).isEqualTo("a:x");
        }
    }
}
