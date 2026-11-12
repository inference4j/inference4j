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

package io.github.inference4j;

import io.github.inference4j.nlp.FlanT5TextGenerator;
import io.github.inference4j.nlp.GlinerEntityRecognizer;
import io.github.inference4j.nlp.NamedEntity;
import io.github.inference4j.nlp.PunctCapSegModel;
import io.github.inference4j.nlp.SentenceTransformerEmbedder;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Function;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * One shared model instance called from many threads must give the same results as sequential calls.
 * Covers a stateless wrapper (shared buffer pool), one with streaming tokenizer state, and a generator
 * (one KV cache per instance, calls serialized).
 */
class ConcurrentInferenceModelTest {

    private static final int THREADS = 8;
    private static final int ROUNDS = 5;
    private static final float TOLERANCE = 1e-5f;
    private static final List<String> TEXTS = List.of(
            "the quick brown fox jumps over the lazy dog",
            "marie curie won the nobel prize in physics and in chemistry",
            "send the invoice to alice at alice@example.com before friday",
            "how do vaccines train the immune system",
            "leonardo da vinci painted the mona lisa in florence",
            "the meeting moved to room four on tuesday afternoon",
            "call john on 555 0100 if the build fails again",
            "rivers carry sediment from the mountains to the sea",
            // Accents, an emoji and CJK split into byte tokens, which a decoder holds across steps
            "josé paid 12 € for a café crème in são paulo 🚀 then flew home",
            "die größte stadt ist berlin und 東京 ist die hauptstadt japans");

    @Test
    void embedderGivesSequentialResultsUnderConcurrency() throws Exception {
        try (var embedder = SentenceTransformerEmbedder.builder().modelId("inference4j/all-MiniLM-L6-v2").build()) {
            List<float[]> sequential = TEXTS.stream().map(embedder::encode).toList();
            for (List<float[]> round : runConcurrently(embedder::encode)) {
                for (int i = 0; i < TEXTS.size(); i++) {
                    assertThat(round.get(i)).containsExactly(sequential.get(i), within(TOLERANCE));
                }
            }
        }
    }

    @Test
    void punctuationModelGivesSequentialResultsUnderConcurrency() throws Exception {
        try (var model = PunctCapSegModel.builder().build()) {
            assertSameAsSequential(model::infer);
        }
    }

    @Test
    void glinerGivesSequentialResultsUnderConcurrency() throws Exception {
        try (var pii = GlinerEntityRecognizer.pii().build()) {
            List<List<NamedEntity>> sequential = TEXTS.stream().map(pii::recognize).toList();
            for (List<List<NamedEntity>> round : runConcurrently(pii::recognize)) {
                for (int i = 0; i < TEXTS.size(); i++) {
                    assertSameEntities(round.get(i), sequential.get(i));
                }
            }
        }
    }

    @Test
    void generatorGivesSequentialResultsUnderConcurrency() throws Exception {
        try (var generator = FlanT5TextGenerator.flanT5Small().maxNewTokens(20).build()) {
            assertSameAsSequential(text -> generator.generate("summarize: " + text).text());
        }
    }

    /** Same spans and labels; scores may differ in the last bits. */
    private static void assertSameEntities(List<NamedEntity> actual, List<NamedEntity> expected) {
        assertThat(actual).hasSameSizeAs(expected);
        for (int i = 0; i < expected.size(); i++) {
            NamedEntity a = actual.get(i);
            NamedEntity e = expected.get(i);
            assertThat(List.of(a.text(), a.label(), a.start(), a.end()))
                    .isEqualTo(List.of(e.text(), e.label(), e.start(), e.end()));
            assertThat(a.score()).isCloseTo(e.score(), within(TOLERANCE));
        }
    }

    private static <R> void assertSameAsSequential(Function<String, R> call) throws Exception {
        List<R> sequential = TEXTS.stream().map(call).toList();
        for (List<R> round : runConcurrently(call)) {
            assertThat(round).containsExactlyElementsOf(sequential);
        }
    }

    /**
     * Runs every text twice through {@code call} from {@value #THREADS} threads at once, for {@value #ROUNDS}
     * rounds; returns the results of each pass over the texts, in text order.
     */
    private static <R> List<List<R>> runConcurrently(Function<String, R> call) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(THREADS);
        try {
            List<List<R>> rounds = new ArrayList<>();
            for (int round = 0; round < ROUNDS; round++) {
                List<Future<R>> first = new ArrayList<>();
                List<Future<R>> second = new ArrayList<>();
                for (String text : TEXTS) {
                    first.add(executor.submit(() -> call.apply(text)));
                    second.add(executor.submit(() -> call.apply(text)));
                }
                List<R> results = new ArrayList<>();
                for (Future<R> future : first) {
                    results.add(future.get());
                }
                rounds.add(results);
                List<R> repeats = new ArrayList<>();
                for (Future<R> future : second) {
                    repeats.add(future.get());
                }
                rounds.add(repeats);
            }
            return rounds;
        } finally {
            executor.shutdownNow();
        }
    }
}
