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

package io.github.inference4j.vision.classification;

import io.github.inference4j.OutputOperator;
import io.github.inference4j.image.Labels;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ResNetTest {

    private static final Labels TEST_LABELS = Labels.of(List.of(
            "cat", "dog", "bird", "fish", "horse"
    ));

    @Test
    void postProcess_returnsTopKWithHighestConfidence() {
        // Logits where "dog" (index 1) is highest, then "horse" (4), then "bird" (2)
        float[] logits = {1.0f, 5.0f, 3.0f, 0.5f, 4.0f};

        List<Classification> results = ResNet.postProcess(logits, TEST_LABELS, 3, OutputOperator.softmax());

        assertEquals(3, results.size());
        assertEquals("dog", results.get(0).label());
        assertEquals(1, results.get(0).index());
        assertEquals("horse", results.get(1).label());
        assertEquals(4, results.get(1).index());
        assertEquals("bird", results.get(2).label());
        assertEquals(2, results.get(2).index());
    }

    @Test
    void postProcess_confidencesSumToLessThanOrEqualOne() {
        float[] logits = {2.0f, 1.0f, 3.0f, 0.5f, 1.5f};

        List<Classification> results = ResNet.postProcess(logits, TEST_LABELS, 5, OutputOperator.softmax());

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
        float[] logits = {0.1f, 0.9f, 0.3f, 0.7f, 0.5f};

        List<Classification> results = ResNet.postProcess(logits, TEST_LABELS, 5, OutputOperator.softmax());

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i - 1).confidence() >= results.get(i).confidence(),
                    "Results not sorted descending at index " + i);
        }
    }

    @Test
    void postProcess_topKLargerThanClasses_returnsAll() {
        float[] logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        List<Classification> results = ResNet.postProcess(logits, TEST_LABELS, 10, OutputOperator.softmax());

        assertEquals(5, results.size());
        assertEquals("horse", results.get(0).label());
    }

    @Test
    void postProcess_topKOne_returnsSingleResult() {
        float[] logits = {0.1f, 0.9f, 0.3f, 0.7f, 0.5f};

        List<Classification> results = ResNet.postProcess(logits, TEST_LABELS, 1, OutputOperator.softmax());

        assertEquals(1, results.size());
        assertEquals("dog", results.get(0).label());
        assertEquals(1, results.get(0).index());
    }

    @Test
    void postProcess_correctIndicesInResult() {
        float[] logits = {10.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        List<Classification> results = ResNet.postProcess(logits, TEST_LABELS, 1, OutputOperator.softmax());

        assertEquals("cat", results.get(0).label());
        assertEquals(0, results.get(0).index());
        assertTrue(results.get(0).confidence() > 0.99f);
    }

    @Test
    void postProcess_equalLogitsGiveEqualConfidence() {
        float[] logits = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        List<Classification> results = ResNet.postProcess(logits, TEST_LABELS, 5, OutputOperator.softmax());

        float expected = 1.0f / 5;
        for (Classification c : results) {
            assertEquals(expected, c.confidence(), 1e-5f);
        }
    }
}
