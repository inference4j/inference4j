package io.github.inference4j.examples;

import io.github.inference4j.embedding.SentenceTransformer;

/**
 * Demonstrates computing semantic similarity between sentence pairs.
 *
 * Requires all-MiniLM-L6-v2 ONNX model — see inference4j-examples/README.md for download instructions.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SemanticSimilarityExample
 */
public class SemanticSimilarityExample {

    public static void main(String[] args) {
        String modelDir = "inference4j-examples/models/all-MiniLM-L6-v2";

        String[][] pairs = {
                {"The cat sat on the mat", "A kitten was resting on the rug"},
                {"The cat sat on the mat", "The stock market crashed yesterday"},
                {"How do I reset my password?", "I forgot my login credentials"},
                {"How do I reset my password?", "What is the weather forecast?"},
                {"Java is a programming language", "Python is used for software development"},
                {"Java is a programming language", "The restaurant serves excellent pasta"},
        };

        try (SentenceTransformer model = SentenceTransformer.fromPretrained(modelDir)) {
            System.out.println("Model loaded successfully.");
            System.out.println("Embedding dimension: " + model.encode("test").length);
            System.out.println();

            for (String[] pair : pairs) {
                float[] embA = model.encode(pair[0]);
                float[] embB = model.encode(pair[1]);
                double similarity = cosineSimilarity(embA, embB);

                System.out.printf("Similarity: %.4f%n", similarity);
                System.out.printf("  A: \"%s\"%n", pair[0]);
                System.out.printf("  B: \"%s\"%n", pair[1]);
                System.out.println();
            }
        }
    }

    static double cosineSimilarity(float[] a, float[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
