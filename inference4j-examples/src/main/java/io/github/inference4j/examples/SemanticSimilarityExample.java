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

package io.github.inference4j.examples;

import io.github.inference4j.nlp.SentenceTransformerEmbedder;

/**
 * Demonstrates computing semantic similarity between sentence pairs.
 *
 * Requires all-MiniLM-L6-v2 ONNX model — see inference4j-examples/README.md for download instructions.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SemanticSimilarityExample
 */
public class SemanticSimilarityExample {

    public static void main(String[] args) {
        String[][] pairs = {
                {"The cat sat on the mat", "A kitten was resting on the rug"},
                {"The cat sat on the mat", "The stock market crashed yesterday"},
                {"How do I reset my password?", "I forgot my login credentials"},
                {"How do I reset my password?", "What is the weather forecast?"},
                {"Java is a programming language", "Python is used for software development"},
                {"Java is a programming language", "The restaurant serves excellent pasta"},
        };

        try (SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .modelId("inference4j/all-MiniLM-L6-v2")
                .build()) {
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
