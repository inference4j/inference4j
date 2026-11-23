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

import io.github.inference4j.model.ModelSource;
import io.github.inference4j.TestModels;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.*;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class MiniLMSearchRerankerTest {

    @TempDir
    Path dir;

    @Test
    void toScore_positiveLogit_highScore() {
        float score = io.github.inference4j.nlp.MiniLMSearchReranker.toScore(5.0f);
        assertThat(score).isGreaterThan(0.99f);
    }

    @Test
    void toScore_negativeLogit_lowScore() {
        float score = MiniLMSearchReranker.toScore(-5.0f);
        assertThat(score).isLessThan(0.01f);
    }

    @Test
    void toScore_zeroLogit_returnsHalf() {
        float score = MiniLMSearchReranker.toScore(0.0f);
        assertThat(score).isCloseTo(0.5f, within(1e-5f));
    }

    @Test
    void toScore_outputBetweenZeroAndOne() {
        float[] testLogits = {-10f, -5f, -1f, 0f, 1f, 5f, 10f};
        for (float logit : testLogits) {
            float score = MiniLMSearchReranker.toScore(logit);
            assertThat(score)
                    .as("Score " + score + " for logit " + logit + " should be in (0, 1)")
                    .isGreaterThan(0f)
                    .isLessThan(1f);
        }
    }

    @Test
    void toScore_monotonicWithLogit() {
        float prev = MiniLMSearchReranker.toScore(-10f);
        for (float logit = -9f; logit <= 10f; logit += 1f) {
            float current = MiniLMSearchReranker.toScore(logit);
            assertThat(current)
                    .as("Score should increase monotonically with logit")
                    .isGreaterThan(prev);
            prev = current;
        }
    }

    @Test
    void toScore_symmetricAroundZero() {
        float scorePlus = MiniLMSearchReranker.toScore(3.0f);
        float scoreMinus = MiniLMSearchReranker.toScore(-3.0f);
        assertThat(scorePlus + scoreMinus).isCloseTo(1.0f, within(1e-5f));
    }

    // --- Builder validation ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThatThrownBy(() ->
                MiniLMSearchReranker.builder()
                        .tokenizer(tokenizer)
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_missingTokenizer_throws() {
        ModelSource source = TestModels.model(dir, "tokens-echo");
        assertThatThrownBy(() ->
                MiniLMSearchReranker.builder()
                        .modelSource(source)
                        .build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("vocab.txt");
    }

    @Test
    void builder_maxLength_passedToTokenizer() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{101, 102},
                        new long[]{1, 1},
                        new long[]{0, 0}));

        try (MiniLMSearchReranker model = MiniLMSearchReranker.builder()
                .modelSource(TestModels.model(dir, "tokens-echo"))
                .tokenizer(tokenizer)
                .maxLength(256)
                .build()) {
            model.score("query", "doc");
        }

        verify(tokenizer).encode("query", "doc", 256);
    }

    // --- Inference flow ---

    @Test
    void score_withTokenTypeIds_returnsCorrectScore() {
        // The fixture model's logit is the sum of the second segment's token ids: 2 + 1 = 3
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{9, 7, 9, 2, 1},
                        new long[]{1, 1, 1, 1, 1},
                        new long[]{0, 0, 0, 1, 1}));

        try (MiniLMSearchReranker model = reranker("tokens-passage-sum", tokenizer)) {
            assertThat(model.score("query", "document")).isCloseTo(sigmoid(3), within(1e-6f));
        }
    }

    @Test
    void score_withoutTokenTypeIds_excludesFromInputs() {
        // The fixture model has no token_type_ids input, which ONNX Runtime would reject; its first logit is the
        // first token id
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{2, 2023, 102, 2003, 102},
                        new long[]{1, 1, 1, 1, 1},
                        new long[]{0, 0, 0, 1, 1}));

        try (MiniLMSearchReranker model = reranker("tokens-echo", tokenizer)) {
            assertThat(model.score("query", "document")).isCloseTo(sigmoid(2), within(1e-6f));
        }
    }

    @Test
    void scoreBatch_returnsScoresForAllDocuments() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(eq("query"), eq("doc1"), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{0, 5, 0, 3, 0},
                        new long[]{1, 1, 1, 1, 1},
                        new long[]{0, 0, 0, 1, 1}));
        when(tokenizer.encode(eq("query"), eq("doc2"), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{0, 5, 0, 0, 0},
                        new long[]{1, 1, 1, 1, 1},
                        new long[]{0, 0, 0, 1, 1}));

        try (MiniLMSearchReranker model = reranker("tokens-passage-sum", tokenizer)) {
            float[] scores = model.scoreBatch("query", List.of("doc1", "doc2"));

            assertThat(scores).containsExactly(new float[]{sigmoid(3), 0.5f}, within(1e-6f));
        }
    }

    @Test
    void strideScoresEveryPassageAndKeepsTheBest() {
        // [CLS] q [SEP] leaves 8 - 3 - 1 = 4 tokens per passage; passages of 4 sharing 2 tokens over the 8
        // document tokens sum to 1, 0 and 2, while the whole document would sum to 3
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("query", Integer.MAX_VALUE)).thenReturn(
                new EncodedInput(new long[]{0, 7, 0}, new long[]{1, 1, 1}, new long[3]));
        when(tokenizer.encode("document", Integer.MAX_VALUE)).thenReturn(
                new EncodedInput(new long[]{0, 1, 0, 0, 0, 0, 0, 0, 2, 0}, new long[10], new long[10]));

        try (MiniLMSearchReranker model = MiniLMSearchReranker.builder()
                .modelSource(TestModels.model(dir, "tokens-passage-sum"))
                .tokenizer(tokenizer)
                .maxLength(8)
                .stride(2)
                .build()) {
            assertThat(model.score("query", "document")).isCloseTo(sigmoid(2), within(1e-6f));
        }
    }

    // --- Close ---

    @Test
    void closedRerankerRejectsScore() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 102}, new long[]{1, 1}, new long[]{0, 0}));
        MiniLMSearchReranker model = reranker("tokens-echo", tokenizer);

        model.close();

        assertThatThrownBy(() -> model.score("query", "doc")).isInstanceOf(IllegalStateException.class);
    }

    private MiniLMSearchReranker reranker(String fixture, Tokenizer tokenizer) {
        return MiniLMSearchReranker.builder()
                .modelSource(TestModels.model(dir, fixture))
                .tokenizer(tokenizer)
                .build();
    }

    private static float sigmoid(double logit) {
        return (float) (1.0 / (1.0 + Math.exp(-logit)));
    }
}
