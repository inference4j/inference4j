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

package io.github.inference4j.embedding;

import io.github.inference4j.routing.RoundRobinRoutingStrategy;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class EmbeddingModelRouterTest {

    static class StubEmbeddingModel implements EmbeddingModel {
        private final float[] embedding;
        boolean closed = false;

        StubEmbeddingModel(float... embedding) {
            this.embedding = embedding;
        }

        @Override
        public float[] encode(String text) {
            return embedding;
        }

        @Override
        public List<float[]> encodeBatch(List<String> texts) {
            return texts.stream().map(t -> embedding).toList();
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    @Test
    void encodeDelegatesToSelectedRoute() {
        StubEmbeddingModel model = new StubEmbeddingModel(1.0f, 2.0f, 3.0f);

        try (EmbeddingModelRouter router = EmbeddingModelRouter.builder()
                .name("test")
                .route("primary", model, 1)
                .build()) {

            assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, router.encode("hello"));
        }
    }

    @Test
    void encodeBatchDelegatesToSelectedRoute() {
        StubEmbeddingModel model = new StubEmbeddingModel(0.5f, 0.5f);

        try (EmbeddingModelRouter router = EmbeddingModelRouter.builder()
                .name("test")
                .route("primary", model, 1)
                .build()) {

            List<float[]> results = router.encodeBatch(List.of("a", "b"));
            assertEquals(2, results.size());
            assertArrayEquals(new float[]{0.5f, 0.5f}, results.get(0));
            assertArrayEquals(new float[]{0.5f, 0.5f}, results.get(1));
        }
    }

    @Test
    void routesWithRoundRobin() {
        StubEmbeddingModel modelA = new StubEmbeddingModel(1.0f);
        StubEmbeddingModel modelB = new StubEmbeddingModel(2.0f);

        try (EmbeddingModelRouter router = EmbeddingModelRouter.builder()
                .name("test")
                .route("a", modelA, 1)
                .route("b", modelB, 1)
                .strategy(new RoundRobinRoutingStrategy())
                .build()) {

            assertArrayEquals(new float[]{1.0f}, router.encode("x"));
            assertArrayEquals(new float[]{2.0f}, router.encode("x"));
            assertArrayEquals(new float[]{1.0f}, router.encode("x"));
        }
    }

    @Test
    void implementsEmbeddingModelInterface() {
        StubEmbeddingModel model = new StubEmbeddingModel(1.0f);

        EmbeddingModel router = EmbeddingModelRouter.builder()
                .name("test")
                .route("primary", model, 1)
                .build();

        // Can be used anywhere an EmbeddingModel is expected
        assertArrayEquals(new float[]{1.0f}, router.encode("hello"));
        router.close();
    }

    @Test
    void closesAllModels() {
        StubEmbeddingModel modelA = new StubEmbeddingModel(1.0f);
        StubEmbeddingModel modelB = new StubEmbeddingModel(2.0f);
        StubEmbeddingModel shadowModel = new StubEmbeddingModel(3.0f);

        EmbeddingModelRouter router = EmbeddingModelRouter.builder()
                .name("test")
                .route("a", modelA, 1)
                .route("b", modelB, 1)
                .shadow("s", shadowModel)
                .build();

        router.close();

        assertTrue(modelA.closed);
        assertTrue(modelB.closed);
        assertTrue(shadowModel.closed);
    }

    @Test
    void shadowDoesNotAffectResult() {
        StubEmbeddingModel primary = new StubEmbeddingModel(1.0f, 2.0f);
        StubEmbeddingModel shadow = new StubEmbeddingModel(9.0f, 9.0f);

        try (EmbeddingModelRouter router = EmbeddingModelRouter.builder()
                .name("test")
                .route("primary", primary, 1)
                .shadow("shadow", shadow)
                .build()) {

            // Result should come from primary, not shadow
            assertArrayEquals(new float[]{1.0f, 2.0f}, router.encode("hello"));
        }
    }
}
