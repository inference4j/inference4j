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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class DistilBertTextClassifierModelTest {

    private DistilBertTextClassifier classifier;

    @BeforeAll
    void setUp() {
        classifier = DistilBertTextClassifier.builder().build();
    }

    @AfterAll
    void tearDown() throws Exception {
        if (classifier != null) classifier.close();
    }

    @Test
    void classify_positiveText_returnsPositiveWithHighConfidence() {
        List<TextClassification> results = classifier.classify("This movie was absolutely fantastic!");

        assertFalse(results.isEmpty(), "Should return classifications");
        assertEquals("POSITIVE", results.get(0).label());
        assertTrue(results.get(0).confidence() > 0.9f,
                "Positive text should have high confidence, got: " + results.get(0).confidence());
    }

    @Test
    void classify_negativeText_returnsNegativeWithHighConfidence() {
        List<TextClassification> results = classifier.classify("This was terrible and boring.");

        assertFalse(results.isEmpty(), "Should return classifications");
        assertEquals("NEGATIVE", results.get(0).label());
        assertTrue(results.get(0).confidence() > 0.9f,
                "Negative text should have high confidence, got: " + results.get(0).confidence());
    }

    @Test
    void classify_topK_limitsResults() {
        List<TextClassification> results = classifier.classify("A decent film.", 1);

        assertEquals(1, results.size(), "topK=1 should return exactly 1 result");
        assertTrue(results.get(0).confidence() > 0f);
        assertTrue(results.get(0).confidence() <= 1f);
    }
}
