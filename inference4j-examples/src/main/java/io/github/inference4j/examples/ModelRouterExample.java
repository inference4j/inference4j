package io.github.inference4j.examples;

import io.github.inference4j.embedding.EmbeddingModel;
import io.github.inference4j.embedding.EmbeddingModelRouter;
import io.github.inference4j.embedding.SentenceTransformer;
import io.github.inference4j.metrics.RouterMetrics;
import io.github.inference4j.routing.RoundRobinRoutingStrategy;

import java.util.ArrayList;
import java.util.List;

/**
 * Demonstrates A/B testing two different models behind a transparent router.
 *
 * <p>Routes between all-MiniLM-L6-v2 (6 layers, faster) and all-MiniLM-L12-v2 (12 layers,
 * potentially higher quality). Both produce 384-dimensional embeddings, so downstream systems
 * work identically regardless of which model serves the request.
 *
 * <p>A custom {@link RouterMetrics} logs which route handled each request with latency — the
 * kind of data you'd combine with user feedback (clicks, conversions) to decide which model wins.
 *
 * <p>Requires both MiniLM models — see inference4j-examples/README.md for download instructions.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ModelRouterExample
 */
public class ModelRouterExample {

    public static void main(String[] args) {
        String l6Dir = "inference4j-examples/models/all-MiniLM-L6-v2";
        String l12Dir = "inference4j-examples/models/all-MiniLM-L12-v2";

        SentenceTransformer l6 = SentenceTransformer.fromPretrained(l6Dir);
        SentenceTransformer l12 = SentenceTransformer.fromPretrained(l12Dir);

        List<String> routeLog = new ArrayList<>();

        RouterMetrics metrics = new RouterMetrics() {
            @Override
            public void recordSuccess(String routerName, String routeName, long durationNanos) {
                routeLog.add(routeName);
                System.out.printf("  [metrics] route=%-5s  duration=%.2fms%n",
                        routeName, durationNanos / 1_000_000.0);
            }

            @Override
            public void recordFailure(String routerName, String routeName, long durationNanos) {
                routeLog.add(routeName + "(failed)");
            }

            @Override
            public void recordShadow(String routerName, String routeName, long durationNanos, boolean success) {}
        };

        try (EmbeddingModel router = EmbeddingModelRouter.builder()
                .name("embedding-ab-test")
                .route("L6", l6, 1)
                .route("L12", l12, 1)
                .strategy(new RoundRobinRoutingStrategy())
                .metrics(metrics)
                .build()) {

            System.out.println("A/B test router loaded (round-robin, 50/50 split).");
            System.out.println("  - L6:  all-MiniLM-L6-v2  (6 layers, faster)");
            System.out.println("  - L12: all-MiniLM-L12-v2 (12 layers, higher quality)");
            System.out.println("  Both produce 384-dimensional embeddings.");
            System.out.println();

            String[] sentences = {
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning models can run efficiently in Java.",
                    "Spring Boot makes it easy to build production applications.",
                    "Transformers revolutionized natural language processing.",
                    "ONNX Runtime enables cross-platform model inference.",
                    "Vector databases power semantic search at scale.",
            };

            for (String sentence : sentences) {
                System.out.printf("encode(\"%s\")%n", sentence);
                float[] embedding = router.encode(sentence);
                System.out.printf("  → %d dimensions%n%n", embedding.length);
            }

            long l6Count = routeLog.stream().filter("L6"::equals).count();
            long l12Count = routeLog.stream().filter("L12"::equals).count();
            System.out.printf("Route distribution: L6=%d, L12=%d%n", l6Count, l12Count);
            System.out.println("In production, combine this with user feedback to pick the winner.");
        }
    }
}
