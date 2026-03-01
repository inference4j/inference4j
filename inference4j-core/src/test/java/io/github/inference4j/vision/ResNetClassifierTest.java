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

package io.github.inference4j.vision;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.OutputOperator;
import io.github.inference4j.processing.Preprocessor;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.preprocessing.image.Labels;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class ResNetClassifierTest {

    private static final Labels TEST_LABELS = Labels.of(List.of(
            "cat", "dog", "bird", "fish", "horse"
    ));

    @Test
    void postProcess_returnsTopKWithHighestConfidence() {
        // Logits where "dog" (index 1) is highest, then "horse" (4), then "bird" (2)
        float[] logits = {1.0f, 5.0f, 3.0f, 0.5f, 4.0f};

        List<Classification> results = io.github.inference4j.vision.ResNetClassifier.postProcess(logits, TEST_LABELS, 3, OutputOperator.softmax());

        assertThat(results).hasSize(3);
        assertThat(results.get(0).label()).isEqualTo("dog");
        assertThat(results.get(0).index()).isEqualTo(1);
        assertThat(results.get(1).label()).isEqualTo("horse");
        assertThat(results.get(1).index()).isEqualTo(4);
        assertThat(results.get(2).label()).isEqualTo("bird");
        assertThat(results.get(2).index()).isEqualTo(2);
    }

    @Test
    void postProcess_confidencesSumToLessThanOrEqualOne() {
        float[] logits = {2.0f, 1.0f, 3.0f, 0.5f, 1.5f};

        List<Classification> results = ResNetClassifier.postProcess(logits, TEST_LABELS, 5, OutputOperator.softmax());

        float sum = 0f;
        for (Classification c : results) {
            assertThat(c.confidence()).isGreaterThan(0f);
            assertThat(c.confidence()).isLessThanOrEqualTo(1f);
            sum += c.confidence();
        }
        assertThat(sum).isCloseTo(1.0f, within(1e-5f));
    }

    @Test
    void postProcess_sortedByConfidenceDescending() {
        float[] logits = {0.1f, 0.9f, 0.3f, 0.7f, 0.5f};

        List<Classification> results = ResNetClassifier.postProcess(logits, TEST_LABELS, 5, OutputOperator.softmax());

        for (int i = 1; i < results.size(); i++) {
            assertThat(results.get(i - 1).confidence())
                    .as("Results not sorted descending at index " + i)
                    .isGreaterThanOrEqualTo(results.get(i).confidence());
        }
    }

    @Test
    void postProcess_topKLargerThanClasses_returnsAll() {
        float[] logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        List<Classification> results = ResNetClassifier.postProcess(logits, TEST_LABELS, 10, OutputOperator.softmax());

        assertThat(results).hasSize(5);
        assertThat(results.get(0).label()).isEqualTo("horse");
    }

    @Test
    void postProcess_topKOne_returnsSingleResult() {
        float[] logits = {0.1f, 0.9f, 0.3f, 0.7f, 0.5f};

        List<Classification> results = ResNetClassifier.postProcess(logits, TEST_LABELS, 1, OutputOperator.softmax());

        assertThat(results).hasSize(1);
        assertThat(results.get(0).label()).isEqualTo("dog");
        assertThat(results.get(0).index()).isEqualTo(1);
    }

    @Test
    void postProcess_correctIndicesInResult() {
        float[] logits = {10.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        List<Classification> results = ResNetClassifier.postProcess(logits, TEST_LABELS, 1, OutputOperator.softmax());

        assertThat(results.get(0).label()).isEqualTo("cat");
        assertThat(results.get(0).index()).isEqualTo(0);
        assertThat(results.get(0).confidence()).isGreaterThan(0.99f);
    }

    @Test
    void postProcess_equalLogitsGiveEqualConfidence() {
        float[] logits = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        List<Classification> results = ResNetClassifier.postProcess(logits, TEST_LABELS, 5, OutputOperator.softmax());

        float expected = 1.0f / 5;
        for (Classification c : results) {
            assertThat(c.confidence()).isCloseTo(expected, within(1e-5f));
        }
    }

    // --- Builder validation ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                ResNetClassifier.builder()
                        .inputName("input")
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_inputNameDefaultsFromSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("data"));

        ResNetClassifier model = ResNetClassifier.builder()
                .session(session)
                .build();

        assertThat(model).isNotNull();
        verify(session).inputNames();
    }

    // --- Inference flow ---

    @Test
    void classify_bufferedImage_returnsCorrectResults() {
        InferenceSession session = mock(InferenceSession.class);
        @SuppressWarnings("unchecked")
        Preprocessor<BufferedImage, Tensor> preprocessor = mock(Preprocessor.class);

        Tensor inputTensor = Tensor.fromFloats(new float[]{0.5f}, new long[]{1});
        when(preprocessor.process(any(BufferedImage.class))).thenReturn(inputTensor);

        Tensor outputTensor = Tensor.fromFloats(
                new float[]{1.0f, 5.0f, 3.0f, 0.5f, 4.0f}, new long[]{1, 5});
        when(session.run(any())).thenReturn(Map.of("output", outputTensor));

        ResNetClassifier model = ResNetClassifier.builder()
                .session(session)
                .preprocessor(preprocessor)
                .labels(TEST_LABELS)
                .inputName("input")
                .build();

        BufferedImage image = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        List<Classification> results = model.classify(image);

        assertThat(results).hasSize(5);
        assertThat(results.get(0).label()).isEqualTo("dog");
        assertThat(results.get(0).index()).isEqualTo(1);
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);

        ResNetClassifier model = ResNetClassifier.builder()
                .session(session)
                .inputName("input")
                .build();

        model.close();

        verify(session).close();
    }
}
