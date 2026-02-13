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

import io.github.inference4j.embedding.SentenceTransformer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Demonstrates semantic search: encode a corpus, then find the most relevant documents for a query.
 *
 * Requires all-MiniLM-L6-v2 ONNX model — see inference4j-examples/README.md for download instructions.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SemanticSearchExample
 */
public class SemanticSearchExample {

    public static void main(String[] args) {
        String modelDir = "inference4j-examples/models/all-MiniLM-L6-v2";

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

        try (SentenceTransformer model = SentenceTransformer.fromPretrained(modelDir)) {
            System.out.println("Model loaded. Encoding corpus...");

            // Encode entire corpus upfront
            List<float[]> corpusEmbeddings = model.encodeBatch(corpus);
            System.out.println("Corpus encoded (" + corpus.size() + " documents).");
            System.out.println();

            for (String query : queries) {
                float[] queryEmbedding = model.encode(query);

                // Score and rank all documents
                List<ScoredDocument> ranked = new ArrayList<>();
                for (int i = 0; i < corpus.size(); i++) {
                    double score = cosineSimilarity(queryEmbedding, corpusEmbeddings.get(i));
                    ranked.add(new ScoredDocument(corpus.get(i), score));
                }
                ranked.sort(Comparator.comparingDouble(ScoredDocument::score).reversed());

                System.out.printf("Query: \"%s\"%n", query);
                System.out.println("Top 3 results:");
                for (int i = 0; i < 3; i++) {
                    ScoredDocument doc = ranked.get(i);
                    System.out.printf("  %d. [%.4f] %s%n", i + 1, doc.score(), doc.text());
                }
                System.out.println();
            }
        }
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
