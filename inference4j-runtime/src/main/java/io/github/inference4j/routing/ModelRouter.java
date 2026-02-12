package io.github.inference4j.routing;

import io.github.inference4j.metrics.NoOpRouterMetrics;
import io.github.inference4j.metrics.RouterMetrics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Abstract base for model routers. Handles route selection, shadow execution,
 * metrics recording, and lifecycle management.
 * <p>
 * Subclasses implement the model interface and delegate each method through
 * {@link #execute(Function)}.
 *
 * @param <T> the model type being routed
 */
public abstract class ModelRouter<T> implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(ModelRouter.class);

    private final String name;
    private final List<Route<T>> routes;
    private final List<Route<T>> shadowRoutes;
    private final RoutingStrategy strategy;
    private final RouterMetrics metrics;
    private final boolean ownsModels;

    protected ModelRouter(BaseBuilder<T, ?> builder) {
        if (builder.routes.isEmpty()) {
            throw new RoutingException("At least one route is required");
        }
        this.name = Objects.requireNonNull(builder.name, "name must not be null");
        this.routes = List.copyOf(builder.routes);
        this.shadowRoutes = List.copyOf(builder.shadowRoutes);
        this.strategy = builder.strategy;
        this.metrics = builder.metrics;
        this.ownsModels = builder.ownsModels;
    }

    /**
     * Executes an operation against a routed model. Handles route selection,
     * timing, metrics, and shadow execution.
     */
    protected <R> R execute(Function<T, R> operation) {
        Route<T> route = strategy.select(routes);

        long startNanos = System.nanoTime();
        R result;
        try {
            result = operation.apply(route.model());
            long durationNanos = System.nanoTime() - startNanos;
            metrics.recordSuccess(name, route.name(), durationNanos);
        } catch (Exception e) {
            long durationNanos = System.nanoTime() - startNanos;
            metrics.recordFailure(name, route.name(), durationNanos);
            throw e;
        }

        executeShadows(operation);

        return result;
    }

    private void executeShadows(Function<T, ?> operation) {
        for (Route<T> shadow : shadowRoutes) {
            long startNanos = System.nanoTime();
            try {
                operation.apply(shadow.model());
                long durationNanos = System.nanoTime() - startNanos;
                metrics.recordShadow(name, shadow.name(), durationNanos, true);
            } catch (Exception e) {
                long durationNanos = System.nanoTime() - startNanos;
                metrics.recordShadow(name, shadow.name(), durationNanos, false);
                log.warn("Shadow route '{}' failed in router '{}': {}", shadow.name(), name, e.getMessage());
            }
        }
    }

    public String name() {
        return name;
    }

    public List<Route<T>> routes() {
        return routes;
    }

    public List<Route<T>> shadowRoutes() {
        return shadowRoutes;
    }

    @Override
    public void close() {
        if (!ownsModels) {
            return;
        }
        for (Route<T> route : routes) {
            closeModel(route);
        }
        for (Route<T> route : shadowRoutes) {
            closeModel(route);
        }
    }

    private void closeModel(Route<T> route) {
        T model = route.model();
        if (model instanceof AutoCloseable closeable) {
            try {
                closeable.close();
            } catch (Exception e) {
                log.warn("Failed to close model for route '{}': {}", route.name(), e.getMessage());
            }
        }
    }

    /**
     * Base builder for model routers.
     */
    protected abstract static class BaseBuilder<T, B extends BaseBuilder<T, B>> {

        private String name;
        private final List<Route<T>> routes = new ArrayList<>();
        private final List<Route<T>> shadowRoutes = new ArrayList<>();
        private RoutingStrategy strategy = new WeightedRoutingStrategy();
        private RouterMetrics metrics = NoOpRouterMetrics.getInstance();
        private boolean ownsModels = true;

        @SuppressWarnings("unchecked")
        private B self() {
            return (B) this;
        }

        public B name(String name) {
            this.name = name;
            return self();
        }

        public B route(String name, T model, int weight) {
            this.routes.add(Route.of(name, model, weight));
            return self();
        }

        public B shadow(String name, T model) {
            this.shadowRoutes.add(Route.shadow(name, model));
            return self();
        }

        public B strategy(RoutingStrategy strategy) {
            this.strategy = Objects.requireNonNull(strategy, "strategy must not be null");
            return self();
        }

        public B metrics(RouterMetrics metrics) {
            this.metrics = Objects.requireNonNull(metrics, "metrics must not be null");
            return self();
        }

        public B ownsModels(boolean ownsModels) {
            this.ownsModels = ownsModels;
            return self();
        }
    }
}
