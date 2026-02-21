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

package io.github.inference4j.nlp;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SentenceTransformerEmbedderModelTest {

    @Test
    void encode_returnsNonEmptyFiniteEmbedding() {
        try (var embedder = io.github.inference4j.nlp.SentenceTransformerEmbedder.builder()
                .modelId("inference4j/all-MiniLM-L6-v2")
                .build()) {

            float[] embedding = embedder.encode("Hello, world!");

            assertTrue(embedding.length > 0, "Embedding should be non-empty");
            for (int i = 0; i < embedding.length; i++) {
                assertTrue(Float.isFinite(embedding[i]),
                        "Embedding value at index " + i + " should be finite, got: " + embedding[i]);
            }
        }
    }

    @Test
    void encodeBatch_returnsMatchingBatchSize() {
        try (var embedder = SentenceTransformerEmbedder.builder()
                .modelId("inference4j/all-MiniLM-L6-v2")
                .build()) {

            List<String> texts = List.of("Hello, world!", "How are you?", "Good morning.");
            List<float[]> embeddings = embedder.encodeBatch(texts);

            assertEquals(3, embeddings.size(), "Should return one embedding per input text");
            int dim = embeddings.get(0).length;
            for (float[] embedding : embeddings) {
                assertEquals(dim, embedding.length, "All embeddings should have the same dimension");
            }
        }
    }

    @Test
    void encode_similarTextProducesCloserEmbeddings() {
        try (var embedder = SentenceTransformerEmbedder.builder()
                .modelId("inference4j/all-MiniLM-L6-v2")
                .build()) {

            float[] embA = embedder.encode("The cat sat on the mat.");
            float[] embB = embedder.encode("A cat is sitting on a mat.");
            float[] embC = embedder.encode("Stock prices rose sharply today.");

            float simAB = cosineSimilarity(embA, embB);
            float simAC = cosineSimilarity(embA, embC);

            assertTrue(simAB > simAC,
                    "Similar sentences should be closer: simAB=" + simAB + " simAC=" + simAC);
        }
    }

    private static float cosineSimilarity(float[] a, float[] b) {
        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
