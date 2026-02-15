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
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.text.ModelConfig;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.nio.file.Path;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
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

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, SENTIMENT_CONFIG, 2, OutputOperator.softmax());

        assertEquals(2, results.size());
        assertEquals("POSITIVE", results.get(0).label());
        assertEquals(1, results.get(0).index());
        assertTrue(results.get(0).confidence() > 0.99f);
    }

    @Test
    void postProcess_sentimentNegative() {
        float[] logits = {4.0f, -1.0f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, SENTIMENT_CONFIG, 2, OutputOperator.softmax());

        assertEquals("NEGATIVE", results.get(0).label());
        assertEquals(0, results.get(0).index());
        assertTrue(results.get(0).confidence() > 0.99f);
    }

    @Test
    void postProcess_returnsTopKSorted() {
        float[] logits = {1.0f, 5.0f, 3.0f, 0.5f, 4.0f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, MULTI_CLASS_CONFIG, 3, OutputOperator.softmax());

        assertEquals(3, results.size());
        assertEquals("joy", results.get(0).label());
        assertEquals(1, results.get(0).index());
        assertEquals("fear", results.get(1).label());
        assertEquals(4, results.get(1).index());
        assertEquals("sadness", results.get(2).label());
        assertEquals(2, results.get(2).index());
    }

    @Test
    void postProcess_softmaxProbabilitiesSumToOne() {
        float[] logits = {2.0f, 1.0f, 3.0f, 0.5f, 1.5f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, MULTI_CLASS_CONFIG, 5, OutputOperator.softmax());

        float sum = 0f;
        for (TextClassification c : results) {
            assertTrue(c.confidence() > 0f);
            assertTrue(c.confidence() <= 1f);
            sum += c.confidence();
        }
        assertEquals(1.0f, sum, 1e-5f);
    }

    @Test
    void postProcess_topKOne_returnsSingleResult() {
        float[] logits = {0.1f, 0.9f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, SENTIMENT_CONFIG, 1, OutputOperator.softmax());

        assertEquals(1, results.size());
        assertEquals("POSITIVE", results.get(0).label());
    }

    @Test
    void postProcess_sigmoidForMultiLabel() {
        float[] logits = {3.0f, -3.0f, 2.0f, -1.0f, 0.0f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, MULTI_CLASS_CONFIG, 5, OutputOperator.sigmoid());

        // Sigmoid doesn't sum to 1 — each class is independent
        assertEquals("anger", results.get(0).label());
        assertTrue(results.get(0).confidence() > 0.9f);
        assertEquals("sadness", results.get(1).label());
        assertTrue(results.get(1).confidence() > 0.8f);
    }

    @Test
    void postProcess_equalLogitsGiveEqualConfidence() {
        float[] logits = {1.0f, 1.0f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, SENTIMENT_CONFIG, 2, OutputOperator.softmax());

        assertEquals(results.get(0).confidence(), results.get(1).confidence(), 1e-5f);
        assertEquals(0.5f, results.get(0).confidence(), 1e-5f);
    }

    @Test
    void postProcess_sortedByConfidenceDescending() {
        float[] logits = {0.1f, 0.9f, 0.3f, 0.7f, 0.5f};

        List<TextClassification> results = DistilBertTextClassifier.postProcess(
                logits, MULTI_CLASS_CONFIG, 5, OutputOperator.softmax());

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i - 1).confidence() >= results.get(i).confidence(),
                    "Results not sorted descending at index " + i);
        }
    }

    // --- Builder validation ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThrows(ModelSourceException.class, () ->
                DistilBertTextClassifier.builder()
                        .tokenizer(tokenizer)
                        .config(SENTIMENT_CONFIG)
                        .modelSource(badSource)
                        .build());
    }

    @Test
    void builder_missingTokenizer_throws() {
        InferenceSession session = mock(InferenceSession.class);
        assertThrows(IllegalStateException.class, () ->
                DistilBertTextClassifier.builder()
                        .session(session)
                        .config(SENTIMENT_CONFIG)
                        .build());
    }

    @Test
    void builder_missingConfig_throws() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThrows(IllegalStateException.class, () ->
                DistilBertTextClassifier.builder()
                        .session(session)
                        .tokenizer(tokenizer)
                        .build());
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

        assertEquals(2, results.size());
        assertEquals("POSITIVE", results.get(0).label());
        assertTrue(results.get(0).confidence() > 0.99f);

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(session).run(captor.capture());
        assertTrue(captor.getValue().containsKey("token_type_ids"));
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
        assertFalse(captor.getValue().containsKey("token_type_ids"));
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
}
