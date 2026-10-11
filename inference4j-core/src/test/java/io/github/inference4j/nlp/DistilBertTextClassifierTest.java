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
import io.github.inference4j.processing.OutputOperator;
import io.github.inference4j.processing.WindowAggregation;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.preprocessing.text.ModelConfig;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.nio.file.Path;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class DistilBertTextClassifierTest {

    private static final ModelConfig SENTIMENT_CONFIG = ModelConfig.of(
            Map.of(0, "NEGATIVE", 1, "POSITIVE"),
            "single_label_classification"
    );

    private static final ModelConfig MULTI_CLASS_CONFIG = ModelConfig.of(
            Map.of(0, "anger", 1, "joy", 2, "sadness", 3, "surprise", 4, "fear"),
            "single_label_classification"
    );

    @Test
    void postProcess_sentimentPositive() {
        // Logits where POSITIVE (index 1) is strongly favored
        float[] logits = {-2.0f, 3.0f};

        List<TextClassification> results = io.github.inference4j.nlp.DistilBertTextClassifier.postProcess(
                logits, SENTIMENT_CONFIG, 2, OutputOperator.softmax());

        assertThat(results).hasSize(2);
        assertThat(results.get(0).label()).isEqualTo("POSITIVE");
        assertThat(results.get(0).index()).isEqualTo(1);
        assertThat(results.get(0).confidence()).isGreaterThan(0.99f);
    }

    @Test
    void postProcess_sentimentNegative() {
        float[] logits = {4.0f, -1.0f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, SENTIMENT_CONFIG, 2, OutputOperator.softmax());

        assertThat(results.get(0).label()).isEqualTo("NEGATIVE");
        assertThat(results.get(0).index()).isEqualTo(0);
        assertThat(results.get(0).confidence()).isGreaterThan(0.99f);
    }

    @Test
    void postProcess_returnsTopKSorted() {
        float[] logits = {1.0f, 5.0f, 3.0f, 0.5f, 4.0f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, MULTI_CLASS_CONFIG, 3, OutputOperator.softmax());

        assertThat(results).hasSize(3);
        assertThat(results.get(0).label()).isEqualTo("joy");
        assertThat(results.get(0).index()).isEqualTo(1);
        assertThat(results.get(1).label()).isEqualTo("fear");
        assertThat(results.get(1).index()).isEqualTo(4);
        assertThat(results.get(2).label()).isEqualTo("sadness");
        assertThat(results.get(2).index()).isEqualTo(2);
    }

    @Test
    void postProcess_softmaxProbabilitiesSumToOne() {
        float[] logits = {2.0f, 1.0f, 3.0f, 0.5f, 1.5f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, MULTI_CLASS_CONFIG, 5, OutputOperator.softmax());

        float sum = 0f;
        for (TextClassification c : results) {
            assertThat(c.confidence()).isGreaterThan(0f);
            assertThat(c.confidence()).isLessThanOrEqualTo(1f);
            sum += c.confidence();
        }
        assertThat(sum).isCloseTo(1.0f, within(1e-5f));
    }

    @Test
    void postProcess_topKOne_returnsSingleResult() {
        float[] logits = {0.1f, 0.9f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, SENTIMENT_CONFIG, 1, OutputOperator.softmax());

        assertThat(results).hasSize(1);
        assertThat(results.get(0).label()).isEqualTo("POSITIVE");
    }

    @Test
    void postProcess_sigmoidForMultiLabel() {
        float[] logits = {3.0f, -3.0f, 2.0f, -1.0f, 0.0f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, MULTI_CLASS_CONFIG, 5, OutputOperator.sigmoid());

        // Sigmoid doesn't sum to 1 — each class is independent
        assertThat(results.get(0).label()).isEqualTo("anger");
        assertThat(results.get(0).confidence()).isGreaterThan(0.9f);
        assertThat(results.get(1).label()).isEqualTo("sadness");
        assertThat(results.get(1).confidence()).isGreaterThan(0.8f);
    }

    @Test
    void postProcess_equalLogitsGiveEqualConfidence() {
        float[] logits = {1.0f, 1.0f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, SENTIMENT_CONFIG, 2, OutputOperator.softmax());

        assertThat(results.get(0).confidence()).isCloseTo(results.get(1).confidence(), within(1e-5f));
        assertThat(results.get(0).confidence()).isCloseTo(0.5f, within(1e-5f));
    }

    @Test
    void postProcess_sortedByConfidenceDescending() {
        float[] logits = {0.1f, 0.9f, 0.3f, 0.7f, 0.5f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, MULTI_CLASS_CONFIG, 5, OutputOperator.softmax());

        for (int i = 1; i < results.size(); i++) {
            assertThat(results.get(i - 1).confidence())
                    .as("Results not sorted descending at index " + i)
                    .isGreaterThanOrEqualTo(results.get(i).confidence());
        }
    }

    // --- Builder validation ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThatThrownBy(() ->
                DistilBertTextClassifier.builder()
                        .tokenizer(tokenizer)
                        .config(SENTIMENT_CONFIG)
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_missingTokenizer_throws() {
        InferenceSession session = mock(InferenceSession.class);
        assertThatThrownBy(() ->
                DistilBertTextClassifier.builder()
                        .session(session)
                        .config(SENTIMENT_CONFIG)
                        .build())
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void builder_missingConfig_throws() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThatThrownBy(() ->
                DistilBertTextClassifier.builder()
                        .session(session)
                        .tokenizer(tokenizer)
                        .build())
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void builder_maxLength_passedToTokenizer() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 102}, new long[]{1, 1}, new long[]{0, 0}));
        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(new float[]{1.0f, -1.0f}, new long[]{1, 2})));

        DistilBertTextClassifier model = DistilBertTextClassifier.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(SENTIMENT_CONFIG)
                .maxLength(128)
                .build();

        model.classify("test");

        verify(tokenizer).encode("test", 128);
    }

    @Test
    void builder_customOutputOperator_applied() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 102}, new long[]{1, 1}, new long[]{0, 0}));
        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(new float[]{3.0f, -3.0f}, new long[]{1, 2})));

        DistilBertTextClassifier model = DistilBertTextClassifier.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(SENTIMENT_CONFIG)
                .outputOperator(OutputOperator.sigmoid())
                .build();

        List<TextClassification> results = model.classify("test");

        // Sigmoid: each class is independent (doesn't sum to 1)
        assertThat(results).hasSize(2);
        assertThat(results.get(0).confidence()).isGreaterThan(0.9f);
    }

    @Test
    void builder_modelIdAndModelSource_accepted() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        DistilBertTextClassifier model = DistilBertTextClassifier.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(SENTIMENT_CONFIG)
                .modelId("custom/model")
                .modelSource(id -> Path.of("/tmp"))
                .build();

        assertThat(model).isNotNull();
    }

    @Test
    void builder_sessionOptions_accepted() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        DistilBertTextClassifier model = DistilBertTextClassifier.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(SENTIMENT_CONFIG)
                .sessionOptions(opts -> { })
                .build();

        assertThat(model).isNotNull();
    }

    // --- Inference flow ---

    @Test
    void classify_withTokenTypeIds_includesInInputs() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask", "token_type_ids"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 2023, 102}, new long[]{1, 1, 1}, new long[]{0, 0, 0}));
        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(new float[]{-2.0f, 3.0f}, new long[]{1, 2})));

        DistilBertTextClassifier model = DistilBertTextClassifier.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(SENTIMENT_CONFIG)
                .build();

        List<TextClassification> results = model.classify("great movie");

        assertThat(results).hasSize(2);
        assertThat(results.get(0).label()).isEqualTo("POSITIVE");
        assertThat(results.get(0).confidence()).isGreaterThan(0.99f);

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(session).run(captor.capture());
        assertThat(captor.getValue()).containsKey("token_type_ids");
    }

    @Test
    void classify_withoutTokenTypeIds_excludesFromInputs() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 2023, 102}, new long[]{1, 1, 1}, new long[]{0, 0, 0}));
        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(new float[]{-2.0f, 3.0f}, new long[]{1, 2})));

        DistilBertTextClassifier model = DistilBertTextClassifier.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(SENTIMENT_CONFIG)
                .build();

        model.classify("great movie");

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(session).run(captor.capture());
        assertThat(captor.getValue()).doesNotContainKey("token_type_ids");
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        DistilBertTextClassifier model = DistilBertTextClassifier.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(SENTIMENT_CONFIG)
                .build();

        model.close();

        verify(session).close();
    }

    // --- long input in windows (stride) ---
    // maxLength 6 leaves 4 content tokens per window; 10 content tokens with stride 2 give
    // windows starting at 0, 2, 4 and 6. Identity output keeps the "probabilities" readable.

    private static final long[] LONG_IDS = {101, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 102};

    private InferenceSession windowSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(session.run(any())).thenReturn(
                logits(0.2f, 0.8f), logits(0.6f, 0.4f), logits(0.1f, 0.9f), logits(0.3f, 0.7f));
        return session;
    }

    private Tokenizer longTokenizer() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("long text", Integer.MAX_VALUE)).thenReturn(encoded(LONG_IDS));
        return tokenizer;
    }

    private DistilBertTextClassifier.Builder windowed(InferenceSession session, Tokenizer tokenizer) {
        return DistilBertTextClassifier.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(SENTIMENT_CONFIG)
                .outputOperator(OutputOperator.identity())
                .maxLength(6)
                .stride(2);
    }

    @Test
    @SuppressWarnings("unchecked")
    void strideClassifiesEveryWindowOfALongText() {
        InferenceSession session = windowSession();

        windowed(session, longTokenizer()).build().classify("long text");

        ArgumentCaptor<Map<String, Tensor>> inputs = ArgumentCaptor.forClass(Map.class);
        verify(session, times(4)).run(inputs.capture());
        List<long[]> windows = new ArrayList<>();
        inputs.getAllValues().forEach(in -> windows.add(in.get("input_ids").toLongs()));
        assertThat(windows).containsExactly(
                new long[]{101, 1, 2, 3, 4, 102},
                new long[]{101, 3, 4, 5, 6, 102},
                new long[]{101, 5, 6, 7, 8, 102},
                new long[]{101, 7, 8, 9, 10, 102});
    }

    @Test
    void strideAveragesWindowScoresByDefault() {
        List<TextClassification> results = windowed(windowSession(), longTokenizer()).build().classify("long text");

        assertThat(results).extracting(TextClassification::label).containsExactly("POSITIVE", "NEGATIVE");
        assertThat(results.get(0).confidence()).isCloseTo(0.7f, within(1e-6f));
        assertThat(results.get(1).confidence()).isCloseTo(0.3f, within(1e-6f));
    }

    @Test
    void maxAggregationKeepsTheStrongestWindowPerLabel() {
        List<TextClassification> results = windowed(windowSession(), longTokenizer())
                .windowAggregation(WindowAggregation.max())
                .build()
                .classify("long text");

        assertThat(results.get(0).confidence()).isCloseTo(0.9f, within(1e-6f));   // POSITIVE, window 3
        assertThat(results.get(1).confidence()).isCloseTo(0.6f, within(1e-6f));   // NEGATIVE, window 2
    }

    @Test
    void customAggregationReceivesEveryWindowInOrder() {
        List<float[]> received = new ArrayList<>();
        WindowAggregation firstWindow = windowScores -> {
            received.addAll(windowScores);
            return windowScores.get(0);
        };

        List<TextClassification> results = windowed(windowSession(), longTokenizer())
                .windowAggregation(firstWindow)
                .build()
                .classify("long text", 1);

        assertThat(received).containsExactly(
                new float[]{0.2f, 0.8f}, new float[]{0.6f, 0.4f}, new float[]{0.1f, 0.9f}, new float[]{0.3f, 0.7f});
        assertThat(results).singleElement().satisfies(r -> {
            assertThat(r.label()).isEqualTo("POSITIVE");
            assertThat(r.confidence()).isCloseTo(0.8f, within(1e-6f));
        });
    }

    @Test
    void strideLeavesInputThatFitsToASinglePass() {
        InferenceSession session = windowSession();
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("short", Integer.MAX_VALUE)).thenReturn(encoded(new long[]{101, 1, 2, 102}));
        when(tokenizer.encode("short", 6)).thenReturn(encoded(new long[]{101, 1, 2, 102}));

        List<TextClassification> results = windowed(session, tokenizer).build().classify("short");

        verify(session, times(1)).run(any());
        assertThat(results.get(0).confidence()).isCloseTo(0.8f, within(1e-6f));
    }

    @Test
    void windowAggregationWithoutStrideIsRejected() {
        assertThatThrownBy(() -> DistilBertTextClassifier.builder()
                .session(mock(InferenceSession.class))
                .tokenizer(mock(Tokenizer.class))
                .config(SENTIMENT_CONFIG)
                .windowAggregation(WindowAggregation.max())
                .build())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("stride");
    }

    @Test
    void strideMustLeaveRoomInTheWindow() {
        assertThatThrownBy(() -> windowed(mock(InferenceSession.class), mock(Tokenizer.class)).stride(4).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("[0, 4)");
    }

    private static Map<String, Tensor> logits(float... values) {
        return Map.of("logits", Tensor.fromFloats(values, new long[]{1, values.length}));
    }

    private static EncodedInput encoded(long[] ids) {
        long[] ones = new long[ids.length];
        java.util.Arrays.fill(ones, 1L);
        return new EncodedInput(ids, ones, new long[ids.length]);
    }
}
