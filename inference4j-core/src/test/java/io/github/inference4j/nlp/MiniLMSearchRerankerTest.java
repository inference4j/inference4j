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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.nio.file.Path;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class MiniLMSearchRerankerTest {

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
        InferenceSession session = mock(InferenceSession.class);
        assertThatThrownBy(() ->
                MiniLMSearchReranker.builder()
                        .session(session)
                        .build())
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void builder_maxLength_passedToTokenizer() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{101, 102},
                        new long[]{1, 1},
                        new long[]{0, 0}));
        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(new float[]{1.0f}, new long[]{1, 1})));

        MiniLMSearchReranker model = MiniLMSearchReranker.builder()
                .session(session)
                .tokenizer(tokenizer)
                .maxLength(256)
                .build();

        model.score("query", "doc");

        verify(tokenizer).encode("query", "doc", 256);
    }

    @Test
    void builder_modelIdAndModelSource_accepted() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        MiniLMSearchReranker model = MiniLMSearchReranker.builder()
                .session(session)
                .tokenizer(tokenizer)
                .modelId("custom/reranker")
                .modelSource(id -> Path.of("/tmp"))
                .build();

        assertThat(model).isNotNull();
    }

    @Test
    void builder_sessionOptions_accepted() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        MiniLMSearchReranker model = MiniLMSearchReranker.builder()
                .session(session)
                .tokenizer(tokenizer)
                .sessionOptions(opts -> { })
                .build();

        assertThat(model).isNotNull();
    }

    // --- Inference flow ---

    @Test
    void score_withTokenTypeIds_returnsCorrectScore() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask", "token_type_ids"));
        when(tokenizer.encode(anyString(), anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{101, 2023, 102, 2003, 102},
                        new long[]{1, 1, 1, 1, 1},
                        new long[]{0, 0, 0, 1, 1}));
        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(new float[]{2.5f}, new long[]{1, 1})));

        MiniLMSearchReranker model = MiniLMSearchReranker.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        float score = model.score("query", "document");

        float expectedScore = (float) (1.0 / (1.0 + Math.exp(-2.5)));
        assertThat(score).isCloseTo(expectedScore, within(1e-5f));

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(session).run(captor.capture());
        assertThat(captor.getValue()).containsKey("token_type_ids");
    }

    @Test
    void score_withoutTokenTypeIds_excludesFromInputs() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{101, 2023, 102, 2003, 102},
                        new long[]{1, 1, 1, 1, 1},
                        new long[]{0, 0, 0, 1, 1}));
        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(new float[]{-1.0f}, new long[]{1, 1})));

        MiniLMSearchReranker model = MiniLMSearchReranker.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        model.score("query", "document");

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(session).run(captor.capture());
        assertThat(captor.getValue()).doesNotContainKey("token_type_ids");
    }

    @Test
    void scoreBatch_returnsScoresForAllDocuments() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{101, 2023, 102, 2003, 102},
                        new long[]{1, 1, 1, 1, 1},
                        new long[]{0, 0, 0, 1, 1}));
        when(session.run(any()))
                .thenReturn(Map.of("logits", Tensor.fromFloats(new float[]{2.5f}, new long[]{1, 1})))
                .thenReturn(Map.of("logits", Tensor.fromFloats(new float[]{-1.0f}, new long[]{1, 1})));

        MiniLMSearchReranker model = MiniLMSearchReranker.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        float[] scores = model.scoreBatch("query", List.of("doc1", "doc2"));

        assertThat(scores).hasSize(2);
        assertThat(scores[0]).isGreaterThan(0.5f);
        assertThat(scores[1]).isLessThan(0.5f);
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        MiniLMSearchReranker model = MiniLMSearchReranker.builder()
                .session(session)
                .tokenizer(tokenizer)
                .build();

        model.close();

        verify(session).close();
    }
}
