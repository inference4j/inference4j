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
import io.github.inference4j.processing.OutputOperator;
import io.github.inference4j.processing.WindowAggregation;
import io.github.inference4j.TestModels;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.preprocessing.text.ModelConfig;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.*;
import static org.assertj.core.api.Assertions.within;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class DistilBertTextClassifierTest {

    @TempDir
    Path dir;

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
        ModelSource source = TestModels.model(dir, "tokens-echo");
        assertThatThrownBy(() ->
                DistilBertTextClassifier.builder()
                        .modelSource(source)
                        .config(SENTIMENT_CONFIG)
                        .build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("vocab.txt");
    }

    @Test
    void builder_missingConfig_throws() {
        ModelSource source = TestModels.model(dir, "tokens-echo");
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThatThrownBy(() ->
                DistilBertTextClassifier.builder()
                        .modelSource(source)
                        .tokenizer(tokenizer)
                        .build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("config.json");
    }

    // The fixture models return one logit per token: its id (+ 1000 x its token type when the model takes
    // token_type_ids), so a two-token input gives the two sentiment logits.

    @Test
    void builder_maxLength_passedToTokenizer() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{101, 102}, new long[]{1, 1}, new long[]{0, 0}));

        try (DistilBertTextClassifier model = sentiment("tokens-echo", tokenizer).maxLength(128).build()) {
            model.classify("test");
        }

        verify(tokenizer).encode("test", 128);
    }

    @Test
    void builder_customOutputOperator_applied() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{3, 0}, new long[]{1, 1}, new long[]{0, 0}));

        try (DistilBertTextClassifier model = sentiment("tokens-echo", tokenizer)
                .outputOperator(OutputOperator.sigmoid())
                .build()) {
            List<TextClassification> results = model.classify("test");

            // Sigmoid: each class is independent (doesn't sum to 1)
            assertThat(results).hasSize(2);
            assertThat(results.get(0).confidence()).isCloseTo(0.9526f, within(1e-4f));
            assertThat(results.get(1).confidence()).isCloseTo(0.5f, within(1e-6f));
        }
    }

    // --- Inference flow ---

    @Test
    void classify_withTokenTypeIds_includesInInputs() {
        // A model without token_type_ids would reject them, and one with them requires them
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{1, 6}, new long[]{1, 1}, new long[]{0, 0}));

        try (DistilBertTextClassifier model = sentiment("tokens-echo-types", tokenizer).build()) {
            List<TextClassification> results = model.classify("great movie");

            assertThat(results).hasSize(2);
            assertThat(results.get(0).label()).isEqualTo("POSITIVE");
            assertThat(results.get(0).confidence()).isGreaterThan(0.99f);
        }
    }

    @Test
    void classify_withoutTokenTypeIds_excludesFromInputs() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{1, 6}, new long[]{1, 1}, new long[]{0, 0}));

        try (DistilBertTextClassifier model = sentiment("tokens-echo", tokenizer).build()) {
            assertThat(model.classify("great movie").get(0).label()).isEqualTo("POSITIVE");
        }
    }

    // --- Close ---

    @Test
    void closedClassifierRejectsClassify() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{1, 6}, new long[]{1, 1}, new long[]{0, 0}));
        DistilBertTextClassifier model = sentiment("tokens-echo", tokenizer).build();

        model.close();

        assertThatThrownBy(() -> model.classify("great movie")).isInstanceOf(IllegalStateException.class);
    }

    private DistilBertTextClassifier.Builder sentiment(String fixture, Tokenizer tokenizer) {
        return DistilBertTextClassifier.builder()
                .modelSource(TestModels.model(dir, fixture))
                .tokenizer(tokenizer)
                .config(SENTIMENT_CONFIG);
    }

    // --- long input in windows (stride) ---
    // maxLength 6 leaves 4 content tokens per window; 10 content tokens with stride 2 give
    // windows starting at 0, 2, 4 and 6. The fixture model returns each window's token ids as its six logits,
    // one per label L0..L5, and the identity output keeps them as the "probabilities".

    private static final long[] LONG_IDS = {101, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 102};

    private static final ModelConfig WINDOW_CONFIG = ModelConfig.of(labelMap(TestModels.labels(6)),
            "single_label_classification");

    private Tokenizer longTokenizer() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("long text", Integer.MAX_VALUE)).thenReturn(encoded(LONG_IDS));
        return tokenizer;
    }

    private DistilBertTextClassifier.Builder windowed(Tokenizer tokenizer) {
        return DistilBertTextClassifier.builder()
                .modelSource(TestModels.model(dir, "tokens-echo"))
                .tokenizer(tokenizer)
                .config(WINDOW_CONFIG)
                .outputOperator(OutputOperator.identity())
                .maxLength(6)
                .stride(2);
    }

    @Test
    void strideClassifiesEveryWindowOfALongText() {
        List<float[]> windows = new ArrayList<>();
        WindowAggregation recording = windowScores -> {
            windows.addAll(windowScores);
            return windowScores.get(0);
        };

        try (DistilBertTextClassifier model = windowed(longTokenizer()).windowAggregation(recording).build()) {
            model.classify("long text");
        }

        assertThat(windows).containsExactly(
                new float[]{101, 1, 2, 3, 4, 102},
                new float[]{101, 3, 4, 5, 6, 102},
                new float[]{101, 5, 6, 7, 8, 102},
                new float[]{101, 7, 8, 9, 10, 102});
    }

    @Test
    void strideAveragesWindowScoresByDefault() {
        try (DistilBertTextClassifier model = windowed(longTokenizer()).build()) {
            List<TextClassification> results = model.classify("long text", 6);

            assertThat(results).extracting(TextClassification::label)
                    .containsExactly("L5", "L0", "L4", "L3", "L2", "L1");
            assertThat(confidences(results)).containsExactly(new float[]{102, 101, 7, 6, 5, 4}, within(1e-4f));
        }
    }

    @Test
    void maxAggregationKeepsTheStrongestWindowPerLabel() {
        try (DistilBertTextClassifier model = windowed(longTokenizer())
                .windowAggregation(WindowAggregation.max())
                .build()) {
            List<TextClassification> results = model.classify("long text", 6);

            assertThat(results).extracting(TextClassification::label)
                    .containsExactly("L5", "L0", "L4", "L3", "L2", "L1");
            assertThat(confidences(results)).containsExactly(new float[]{102, 101, 10, 9, 8, 7}, within(1e-4f));
        }
    }

    @Test
    void customAggregationDecidesTheScores() {
        WindowAggregation secondWindow = windowScores -> windowScores.get(1);

        try (DistilBertTextClassifier model = windowed(longTokenizer()).windowAggregation(secondWindow).build()) {
            List<TextClassification> results = model.classify("long text", 6);

            assertThat(confidences(results)).containsExactly(new float[]{102, 101, 6, 5, 4, 3}, within(1e-4f));
        }
    }

    @Test
    void strideLeavesInputThatFitsToASinglePass() {
        WindowAggregation unexpected = windowScores -> {
            throw new AssertionError("input that fits must not be split into windows");
        };
        long[] ids = {101, 1, 2, 3, 4, 102};
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode("short", Integer.MAX_VALUE)).thenReturn(encoded(ids));
        when(tokenizer.encode("short", 6)).thenReturn(encoded(ids));

        try (DistilBertTextClassifier model = windowed(tokenizer).windowAggregation(unexpected).build()) {
            List<TextClassification> results = model.classify("short", 6);

            assertThat(confidences(results)).containsExactly(new float[]{102, 101, 4, 3, 2, 1}, within(1e-4f));
        }
    }

    @Test
    void windowAggregationWithoutStrideIsRejected() {
        assertThatThrownBy(() -> DistilBertTextClassifier.builder()
                .tokenizer(mock(Tokenizer.class))
                .config(SENTIMENT_CONFIG)
                .windowAggregation(WindowAggregation.max())
                .build())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("stride");
    }

    @Test
    void strideMustLeaveRoomInTheWindow() {
        assertThatThrownBy(() -> windowed(mock(Tokenizer.class)).stride(4).build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("[0, 4)");
    }

    private static float[] confidences(List<TextClassification> results) {
        float[] confidences = new float[results.size()];
        for (int i = 0; i < confidences.length; i++) {
            confidences[i] = results.get(i).confidence();
        }
        return confidences;
    }

    private static Map<Integer, String> labelMap(String[] labels) {
        Map<Integer, String> map = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            map.put(i, labels[i]);
        }
        return map;
    }

    private static EncodedInput encoded(long[] ids) {
        long[] ones = new long[ids.length];
        Arrays.fill(ones, 1L);
        return new EncodedInput(ids, ones, new long[ids.length]);
    }
}
