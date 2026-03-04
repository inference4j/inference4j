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

import io.github.inference4j.nlp.PoolingStrategy;
import io.github.inference4j.nlp.SentenceTransformerEmbedder;

/**
 * Demonstrates BGE embedding model with CLS pooling and L2 normalization.
 *
 * BGE (BAAI General Embedding) uses CLS pooling instead of MEAN pooling,
 * and recommends L2 normalization for cosine similarity comparisons.
 *
 * Requires bge-base-en-v1.5 ONNX model (~430 MB).
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.BgeEmbeddingExample
 */
public class BgeEmbeddingExample {

    public static void main(String[] args) {
        String[][] pairs = {
                {"What is machine learning?", "ML is a subset of artificial intelligence"},
                {"What is machine learning?", "The recipe calls for two cups of flour"},
                {"How does photosynthesis work?", "Plants convert sunlight into chemical energy"},
                {"How does photosynthesis work?", "The stock market opened higher today"},
        };

        try (SentenceTransformerEmbedder model = SentenceTransformerEmbedder.builder()
                .modelId("inference4j/bge-base-en-v1.5")
                .poolingStrategy(PoolingStrategy.CLS)
                .normalize()
                .build()) {
            System.out.println("BGE Base EN v1.5 loaded successfully.");
            System.out.println("Embedding dimension: " + model.encode("test").length);
            System.out.println("Pooling: CLS, Normalization: L2");
            System.out.println();

            for (String[] pair : pairs) {
                float[] embA = model.encode(pair[0]);
                float[] embB = model.encode(pair[1]);

                // With L2-normalized vectors, cosine similarity = dot product
                double similarity = dotProduct(embA, embB);

                System.out.printf("Similarity: %.4f%n", similarity);
                System.out.printf("  A: \"%s\"%n", pair[0]);
                System.out.printf("  B: \"%s\"%n", pair[1]);
                System.out.println();
            }
        }
    }

    static double dotProduct(float[] a, float[] b) {
        double dot = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }
}
