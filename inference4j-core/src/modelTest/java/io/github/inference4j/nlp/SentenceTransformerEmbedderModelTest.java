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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class SentenceTransformerEmbedderModelTest {

    private SentenceTransformerEmbedder embedder;

    @BeforeAll
    void setUp() {
        embedder = SentenceTransformerEmbedder.builder()
                .modelId("inference4j/all-MiniLM-L6-v2")
                .build();
    }

    @AfterAll
    void tearDown() throws Exception {
        if (embedder != null) embedder.close();
    }

    @Test
    void encode_returnsNonEmptyFiniteEmbedding() {
        float[] embedding = embedder.encode("Hello, world!");

        assertThat(embedding.length > 0).as("Embedding should be non-empty").isTrue();
        for (int i = 0; i < embedding.length; i++) {
            assertThat(Float.isFinite(embedding[i])).as("Embedding value at index " + i + " should be finite, got: " + embedding[i]).isTrue();
        }
    }

    @Test
    void encodeBatch_returnsMatchingBatchSize() {
        List<String> texts = List.of("Hello, world!", "How are you?", "Good morning.");
        List<float[]> embeddings = embedder.encodeBatch(texts);

        assertThat(embeddings.size()).as("Should return one embedding per input text").isEqualTo(3);
        int dim = embeddings.get(0).length;
        for (float[] embedding : embeddings) {
            assertThat(embedding.length).as("All embeddings should have the same dimension").isEqualTo(dim);
        }
    }

    @Test
    void encode_similarTextProducesCloserEmbeddings() {
        float[] embA = embedder.encode("The cat sat on the mat.");
        float[] embB = embedder.encode("A cat is sitting on a mat.");
        float[] embC = embedder.encode("Stock prices rose sharply today.");

        float simAB = cosineSimilarity(embA, embB);
        float simAC = cosineSimilarity(embA, embC);

        assertThat(simAB > simAC).as("Similar sentences should be closer: simAB=" + simAB + " simAC=" + simAC).isTrue();
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
