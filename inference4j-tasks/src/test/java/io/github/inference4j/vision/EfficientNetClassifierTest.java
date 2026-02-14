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
import io.github.inference4j.ModelSource;
import io.github.inference4j.OutputOperator;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.image.ImageTransformPipeline;
import io.github.inference4j.image.Labels;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class EfficientNetClassifierTest {

    private static final Labels TEST_LABELS = Labels.of(List.of(
            "cat", "dog", "bird", "fish", "horse"
    ));

    @Test
    void postProcess_returnsTopKWithHighestProbability() {
        // EfficientNet outputs probabilities (softmax already applied in model)
        float[] probs = {0.05f, 0.50f, 0.20f, 0.05f, 0.20f};

        List<Classification> results = EfficientNetClassifier.postProcess(probs, TEST_LABELS, 3, OutputOperator.identity());

        assertEquals(3, results.size());
        assertEquals("dog", results.get(0).label());
        assertEquals(1, results.get(0).index());
        assertEquals(0.50f, results.get(0).confidence(), 1e-5f);
    }

    @Test
    void postProcess_preservesProbabilitiesWithoutSoftmax() {
        float[] probs = {0.10f, 0.40f, 0.20f, 0.05f, 0.25f};

        List<Classification> results = EfficientNetClassifier.postProcess(probs, TEST_LABELS, 5, OutputOperator.identity());

        float sum = 0f;
        for (Classification c : results) {
            assertTrue(c.confidence() > 0f);
            assertTrue(c.confidence() <= 1f);
            sum += c.confidence();
        }
        assertEquals(1.0f, sum, 1e-5f);
    }

    @Test
    void postProcess_sortedByConfidenceDescending() {
        float[] probs = {0.10f, 0.40f, 0.15f, 0.30f, 0.05f};

        List<Classification> results = EfficientNetClassifier.postProcess(probs, TEST_LABELS, 5, OutputOperator.identity());

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i - 1).confidence() >= results.get(i).confidence(),
                    "Results not sorted descending at index " + i);
        }
    }

    @Test
    void postProcess_topKLargerThanClasses_returnsAll() {
        float[] probs = {0.05f, 0.10f, 0.15f, 0.20f, 0.50f};

        List<Classification> results = EfficientNetClassifier.postProcess(probs, TEST_LABELS, 10, OutputOperator.identity());

        assertEquals(5, results.size());
        assertEquals("horse", results.get(0).label());
    }

    @Test
    void postProcess_topKOne_returnsSingleResult() {
        float[] probs = {0.05f, 0.60f, 0.15f, 0.10f, 0.10f};

        List<Classification> results = EfficientNetClassifier.postProcess(probs, TEST_LABELS, 1, OutputOperator.identity());

        assertEquals(1, results.size());
        assertEquals("dog", results.get(0).label());
        assertEquals(1, results.get(0).index());
    }

    @Test
    void postProcess_highConfidencePreserved() {
        float[] probs = {0.95f, 0.02f, 0.01f, 0.01f, 0.01f};

        List<Classification> results = EfficientNetClassifier.postProcess(probs, TEST_LABELS, 1, OutputOperator.identity());

        assertEquals("cat", results.get(0).label());
        assertEquals(0, results.get(0).index());
        assertTrue(results.get(0).confidence() > 0.90f);
    }

    @Test
    void postProcess_equalProbabilitiesGiveEqualConfidence() {
        float[] probs = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

        List<Classification> results = EfficientNetClassifier.postProcess(probs, TEST_LABELS, 5, OutputOperator.identity());

        for (Classification c : results) {
            assertEquals(0.2f, c.confidence(), 1e-5f);
        }
    }

    // --- Builder validation ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThrows(ModelSourceException.class, () ->
                EfficientNetClassifier.builder()
                        .inputName("images:0")
                        .modelSource(badSource)
                        .build());
    }

    @Test
    void builder_inputNameDefaultsFromSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("images:0"));

        EfficientNetClassifier model = EfficientNetClassifier.builder()
                .session(session)
                .build();

        assertNotNull(model);
        verify(session).inputNames();
    }

    // --- Inference flow ---

    @Test
    void classify_bufferedImage_returnsCorrectResults() {
        InferenceSession session = mock(InferenceSession.class);
        ImageTransformPipeline pipeline = mock(ImageTransformPipeline.class);

        Tensor inputTensor = Tensor.fromFloats(new float[]{0.5f}, new long[]{1});
        when(pipeline.transform(any(BufferedImage.class))).thenReturn(inputTensor);

        Tensor outputTensor = Tensor.fromFloats(
                new float[]{0.05f, 0.50f, 0.20f, 0.05f, 0.20f}, new long[]{1, 5});
        when(session.run(any())).thenReturn(Map.of("output", outputTensor));

        EfficientNetClassifier model = EfficientNetClassifier.builder()
                .session(session)
                .pipeline(pipeline)
                .labels(TEST_LABELS)
                .inputName("images:0")
                .outputOperator(OutputOperator.identity())
                .build();

        BufferedImage image = new BufferedImage(280, 280, BufferedImage.TYPE_INT_RGB);
        List<Classification> results = model.classify(image);

        assertEquals(5, results.size());
        assertEquals("dog", results.get(0).label());
        assertEquals(0.50f, results.get(0).confidence(), 1e-5f);
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);

        EfficientNetClassifier model = EfficientNetClassifier.builder()
                .session(session)
                .inputName("images:0")
                .build();

        model.close();

        verify(session).close();
    }
}
