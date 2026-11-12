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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.inference4j.processing.MathOps;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * Checks nomic-embed-text-v1.5 against embeddings from the upstream PyTorch model, produced by
 * onnx-exporter's {@code validation/nomic_embed_validate.py}.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class NomicEmbedTextModelTest {

    private static final String MODEL = "inference4j/nomic-embed-text-v1.5";
    private static final float MIN_COSINE = 0.9999f;

    private JsonNode reference;

    @BeforeAll
    void loadReference() throws IOException {
        try (InputStream in = getClass().getResourceAsStream("/fixtures/nomic-embed-text-v1.5-reference.json")) {
            reference = new ObjectMapper().readTree(in);
        }
    }

    @Test
    void fullEmbeddingsMatchReference() {
        try (var embedder = SentenceTransformerEmbedder.builder().modelId(MODEL).normalize().build()) {
            assertMatches(embedder, reference.get("embeddings_768"), 768);
        }
    }

    @Test
    void matryoshkaEmbeddingsMatchReference() {
        for (int dimensions : new int[]{256, 64}) {
            // The reference follows the model card (layer norm, truncate, normalize). Pooled vectors are
            // near zero-mean, so plain truncate-and-normalize matches it to a cosine of about 0.99997.
            try (var embedder = SentenceTransformerEmbedder.builder().modelId(MODEL)
                    .dimensions(dimensions).normalize().build()) {
                assertMatches(embedder, reference.get("embeddings_" + dimensions), dimensions);
            }
        }
    }

    @Test
    void longInputMatchesReference() {
        JsonNode longText = reference.get("long_text");
        String text = longText.get("paragraph").asText().repeat(longText.get("repeat").asInt());
        try (var embedder = SentenceTransformerEmbedder.builder().modelId(MODEL)
                .textPrefix(longText.get("prefix").asText()).maxLength(8192).normalize().build()) {
            float[] actual = embedder.encode(text);
            assertThat(MathOps.dotProduct(actual, floats(reference.get("long_embedding_768"))))
                    .isGreaterThan(MIN_COSINE);
        }
    }

    @Test
    void queriesRankTheirDocumentFirst() {
        try (var embedder = SentenceTransformerEmbedder.builder().modelId(MODEL).normalize().build()) {
            float[] capital = embedder.encode("What is the capital of France?", "search_query: ");
            List<float[]> documents = embedder.encodeBatch(List.of(
                    "Paris is the capital and largest city of France.",
                    "The Great Barrier Reef is the world's largest coral reef system."), "search_document: ");

            assertThat(MathOps.dotProduct(capital, documents.get(0)))
                    .isGreaterThan(MathOps.dotProduct(capital, documents.get(1)) + 0.2f);
        }
    }

    @Test
    void perCallPrefixMatchesTheBuilderPrefix() {
        String text = "What is the capital of France?";
        try (var shared = SentenceTransformerEmbedder.builder().modelId(MODEL).normalize().build();
             var queries = SentenceTransformerEmbedder.builder().modelId(MODEL)
                     .textPrefix("search_query: ").normalize().build()) {
            assertThat(shared.encode(text, "search_query: "))
                    .containsExactly(queries.encode(text), within(1e-6f));
        }
    }

    private void assertMatches(SentenceTransformerEmbedder embedder, JsonNode expected, int dimensions) {
        JsonNode texts = reference.get("texts");
        for (int i = 0; i < texts.size(); i++) {
            float[] actual = embedder.encode(texts.get(i).asText());
            assertThat(actual).hasSize(dimensions);
            assertThat(MathOps.dotProduct(actual, floats(expected.get(i))))
                    .as("cosine with reference for \"%s\" at %d dimensions", texts.get(i).asText(), dimensions)
                    .isGreaterThan(MIN_COSINE);
        }
    }

    private static float[] floats(JsonNode array) {
        float[] values = new float[array.size()];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) array.get(i).asDouble();
        }
        return values;
    }
}
