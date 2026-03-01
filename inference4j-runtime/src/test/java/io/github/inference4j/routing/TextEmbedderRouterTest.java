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

package io.github.inference4j.routing;

import io.github.inference4j.nlp.TextEmbedder;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

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

        try (io.github.inference4j.routing.TextEmbedderRouter router = io.github.inference4j.routing.TextEmbedderRouter.builder()
                .name("test")
                .route("primary", model, 1)
                .build()) {

            assertThat(router.encode("hello")).isEqualTo(new float[]{1.0f, 2.0f, 3.0f});
        }
    }

    @Test
    void encodeBatchDelegatesToSelectedRoute() {
        StubTextEmbedder model = new StubTextEmbedder(0.5f, 0.5f);

        try (io.github.inference4j.routing.TextEmbedderRouter router = io.github.inference4j.routing.TextEmbedderRouter.builder()
                .name("test")
                .route("primary", model, 1)
                .build()) {

            List<float[]> results = router.encodeBatch(List.of("a", "b"));
            assertThat(results).hasSize(2);
            assertThat(results.get(0)).isEqualTo(new float[]{0.5f, 0.5f});
            assertThat(results.get(1)).isEqualTo(new float[]{0.5f, 0.5f});
        }
    }

    @Test
    void routesWithRoundRobin() {
        StubTextEmbedder modelA = new StubTextEmbedder(1.0f);
        StubTextEmbedder modelB = new StubTextEmbedder(2.0f);

        try (io.github.inference4j.routing.TextEmbedderRouter router = io.github.inference4j.routing.TextEmbedderRouter.builder()
                .name("test")
                .route("a", modelA, 1)
                .route("b", modelB, 1)
                .strategy(new RoundRobinRoutingStrategy())
                .build()) {

            assertThat(router.encode("x")).isEqualTo(new float[]{1.0f});
            assertThat(router.encode("x")).isEqualTo(new float[]{2.0f});
            assertThat(router.encode("x")).isEqualTo(new float[]{1.0f});
        }
    }

    @Test
    void implementsTextEmbedderInterface() {
        StubTextEmbedder model = new StubTextEmbedder(1.0f);

        TextEmbedder router = io.github.inference4j.routing.TextEmbedderRouter.builder()
                .name("test")
                .route("primary", model, 1)
                .build();

        // Can be used anywhere a TextEmbedder is expected
        assertThat(router.encode("hello")).isEqualTo(new float[]{1.0f});
        router.close();
    }

    @Test
    void closesAllModels() {
        StubTextEmbedder modelA = new StubTextEmbedder(1.0f);
        StubTextEmbedder modelB = new StubTextEmbedder(2.0f);
        StubTextEmbedder shadowModel = new StubTextEmbedder(3.0f);

        io.github.inference4j.routing.TextEmbedderRouter router = io.github.inference4j.routing.TextEmbedderRouter.builder()
                .name("test")
                .route("a", modelA, 1)
                .route("b", modelB, 1)
                .shadow("s", shadowModel)
                .build();

        router.close();

        assertThat(modelA.closed).isTrue();
        assertThat(modelB.closed).isTrue();
        assertThat(shadowModel.closed).isTrue();
    }

    @Test
    void shadowDoesNotAffectResult() {
        StubTextEmbedder primary = new StubTextEmbedder(1.0f, 2.0f);
        StubTextEmbedder shadow = new StubTextEmbedder(9.0f, 9.0f);

        try (io.github.inference4j.routing.TextEmbedderRouter router = io.github.inference4j.routing.TextEmbedderRouter.builder()
                .name("test")
                .route("primary", primary, 1)
                .shadow("shadow", shadow)
                .build()) {

            // Result should come from primary, not shadow
            assertThat(router.encode("hello")).isEqualTo(new float[]{1.0f, 2.0f});
        }
    }
}
