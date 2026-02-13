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

import io.github.inference4j.embedding.EmbeddingModel;
import io.github.inference4j.embedding.SentenceTransformer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Compares semantic search rankings between two embedding models on the same corpus and queries.
 *
 * <p>Each model encodes the corpus independently, then ranks documents for each query.
 * Results are printed side by side so you can see how model choice affects retrieval quality.
 *
 * <p>Requires two ONNX models — see inference4j-examples/README.md for download instructions.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ModelComparisonExample
 */
public class ModelComparisonExample {

    public static void main(String[] args) {
        String miniLmDir = "inference4j-examples/models/all-MiniLM-L6-v2";
        String mpnetDir = "inference4j-examples/models/all-mpnet-base-v2";

        List<String> corpus = List.of(
                "Java is a statically typed, object-oriented programming language.",
                "Python is popular for machine learning and data science.",
                "The Eiffel Tower is located in Paris, France.",
                "ONNX Runtime provides cross-platform model inference.",
                "Spring Boot simplifies building production-ready Java applications.",
                "Transformers use self-attention to process sequential data.",
                "The Great Wall of China is visible from space.",
                "Docker containers package applications with their dependencies.",
                "BERT is a pre-trained language model by Google.",
                "PostgreSQL is an advanced open-source relational database."
        );

        String[] queries = {
                "How do I run ML models in Java?",
                "Tell me about famous landmarks",
                "What are good databases for web apps?"
        };

        try (EmbeddingModel miniLm = SentenceTransformer.fromPretrained(miniLmDir);
             EmbeddingModel mpnet = SentenceTransformer.fromPretrained(mpnetDir)) {

            System.out.println("Models loaded:");
            System.out.println("  A: all-MiniLM-L6-v2 (384 dimensions)");
            System.out.println("  B: all-mpnet-base-v2 (768 dimensions)");
            System.out.println();

            List<float[]> corpusA = miniLm.encodeBatch(corpus);
            List<float[]> corpusB = mpnet.encodeBatch(corpus);

            for (String query : queries) {
                List<ScoredDocument> rankedA = rank(query, corpus, corpusA, miniLm);
                List<ScoredDocument> rankedB = rank(query, corpus, corpusB, mpnet);

                System.out.printf("Query: \"%s\"%n", query);
                System.out.println("  MiniLM-L6-v2              | mpnet-base-v2");
                System.out.println("  --------------------------+---------------------------");
                for (int i = 0; i < 3; i++) {
                    ScoredDocument a = rankedA.get(i);
                    ScoredDocument b = rankedB.get(i);
                    System.out.printf("  %d. [%.4f] %-15s | %d. [%.4f] %-15s%n",
                            i + 1, a.score(), truncate(a.text(), 15),
                            i + 1, b.score(), truncate(b.text(), 15));
                }
                System.out.println();
            }
        }
    }

    private static List<ScoredDocument> rank(String query, List<String> corpus,
                                             List<float[]> corpusEmbeddings, EmbeddingModel model) {
        float[] queryEmbedding = model.encode(query);
        List<ScoredDocument> ranked = new ArrayList<>();
        for (int i = 0; i < corpus.size(); i++) {
            double score = cosineSimilarity(queryEmbedding, corpusEmbeddings.get(i));
            ranked.add(new ScoredDocument(corpus.get(i), score));
        }
        ranked.sort(Comparator.comparingDouble(ScoredDocument::score).reversed());
        return ranked;
    }

    private static String truncate(String text, int maxLen) {
        return text.length() <= maxLen ? text : text.substring(0, maxLen - 1) + "…";
    }

    record ScoredDocument(String text, double score) {}

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
