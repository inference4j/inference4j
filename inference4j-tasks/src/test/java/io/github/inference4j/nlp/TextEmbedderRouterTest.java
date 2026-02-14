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

import io.github.inference4j.routing.RoundRobinRoutingStrategy;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TextEmbedderRouterTest {

    static class StubTextEmbedder implements TextEmbedder {
        private final float[] embedding;
        boolean closed = false;

        StubTextEmbedder(float... embedding) {
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
        StubTextEmbedder model = new StubTextEmbedder(1.0f, 2.0f, 3.0f);

        try (TextEmbedderRouter router = TextEmbedderRouter.builder()
                .name("test")
                .route("primary", model, 1)
                .build()) {

            assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, router.encode("hello"));
        }
    }

    @Test
    void encodeBatchDelegatesToSelectedRoute() {
        StubTextEmbedder model = new StubTextEmbedder(0.5f, 0.5f);

        try (TextEmbedderRouter router = TextEmbedderRouter.builder()
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
        StubTextEmbedder modelA = new StubTextEmbedder(1.0f);
        StubTextEmbedder modelB = new StubTextEmbedder(2.0f);

        try (TextEmbedderRouter router = TextEmbedderRouter.builder()
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
    void implementsTextEmbedderInterface() {
        StubTextEmbedder model = new StubTextEmbedder(1.0f);

        TextEmbedder router = TextEmbedderRouter.builder()
                .name("test")
                .route("primary", model, 1)
                .build();

        // Can be used anywhere a TextEmbedder is expected
        assertArrayEquals(new float[]{1.0f}, router.encode("hello"));
        router.close();
    }

    @Test
    void closesAllModels() {
        StubTextEmbedder modelA = new StubTextEmbedder(1.0f);
        StubTextEmbedder modelB = new StubTextEmbedder(2.0f);
        StubTextEmbedder shadowModel = new StubTextEmbedder(3.0f);

        TextEmbedderRouter router = TextEmbedderRouter.builder()
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
        StubTextEmbedder primary = new StubTextEmbedder(1.0f, 2.0f);
        StubTextEmbedder shadow = new StubTextEmbedder(9.0f, 9.0f);

        try (TextEmbedderRouter router = TextEmbedderRouter.builder()
                .name("test")
                .route("primary", primary, 1)
                .shadow("shadow", shadow)
                .build()) {

            // Result should come from primary, not shadow
            assertArrayEquals(new float[]{1.0f, 2.0f}, router.encode("hello"));
        }
    }
}
