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

import io.github.inference4j.text.MiniLMReranker;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Demonstrates cross-encoder reranking with ms-marco-MiniLM-L-6-v2.
 *
 * <p>Shows a typical two-stage retrieval pipeline: a fast first stage (simulated here)
 * retrieves candidate documents, then the cross-encoder reranks them for precision.
 *
 * Requires ms-marco-MiniLM-L-6-v2 ONNX model (~91 MB).
 * See inference4j-examples/README.md for download instructions.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.CrossEncoderRerankerExample
 */
public class CrossEncoderRerankerExample {

    public static void main(String[] args) {
        String query = "What is the capital of France?";

        // Simulated first-stage retrieval results (e.g., from BM25 or bi-encoder)
        List<String> candidates = List.of(
                "Paris is the capital and largest city of France.",
                "France is a country in Western Europe.",
                "The Eiffel Tower is a famous landmark in Paris.",
                "Berlin is the capital of Germany.",
                "Lyon is the third-largest city in France.",
                "The capital of Italy is Rome.",
                "French cuisine is renowned worldwide.",
                "Paris was founded in the 3rd century BC."
        );

        try (MiniLMReranker reranker = MiniLMReranker.builder().build()) {
            System.out.println("MiniLM reranker loaded successfully.");
            System.out.println();

            // Score all candidates against the query
            float[] scores = reranker.scoreBatch(query, candidates);

            // Rank by score
            List<ScoredDocument> ranked = new ArrayList<>();
            for (int i = 0; i < candidates.size(); i++) {
                ranked.add(new ScoredDocument(candidates.get(i), scores[i]));
            }
            ranked.sort(Comparator.comparingDouble(ScoredDocument::score).reversed());

            System.out.printf("Query: \"%s\"%n", query);
            System.out.println();
            System.out.println("Reranked results:");
            for (int i = 0; i < ranked.size(); i++) {
                ScoredDocument doc = ranked.get(i);
                System.out.printf("  %d. [%.4f] %s%n", i + 1, doc.score(), doc.text());
            }
        }
    }

    record ScoredDocument(String text, float score) {}
}
