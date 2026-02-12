package io.github.inference4j.routing;

import io.github.inference4j.metrics.RouterMetrics;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

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

            assertEquals("result:hello", router.process("hello"));
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

        assertEquals(List.of("success:test:primary"), metrics.events);
    }

    @Test
    void recordsFailureMetrics() {
        RecordingMetrics metrics = new RecordingMetrics();

        try (TestModelRouter router = TestModelRouter.builder()
                .name("test")
                .route("primary", new FailingTestModel(), 1)
                .metrics(metrics)
                .build()) {

            assertThrows(RuntimeException.class, () -> router.process("hello"));
        }

        assertEquals(List.of("failure:test:primary"), metrics.events);
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
            assertEquals("primary:hello", result);
        }

        assertEquals(2, metrics.events.size());
        assertEquals("success:test:primary", metrics.events.get(0));
        assertEquals("shadow:test:shadow:true", metrics.events.get(1));
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
            assertEquals("ok:hello", result);
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

        assertTrue(model1.closed);
        assertTrue(model2.closed);
        assertTrue(shadowModel.closed);
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

        assertFalse(model.closed);
    }

    @Test
    void throwsWhenNoRoutes() {
        assertThrows(RoutingException.class, () ->
                TestModelRouter.builder()
                        .name("test")
                        .build());
    }

    @Test
    void throwsWhenNoName() {
        assertThrows(NullPointerException.class, () ->
                TestModelRouter.builder()
                        .route("r", new StubTestModel("a"), 1)
                        .build());
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

            assertEquals("a:x", router.process("x"));
            assertEquals("b:x", router.process("x"));
            assertEquals("a:x", router.process("x"));
        }
    }
}
