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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ResNetClassifierModelTest {

    private ResNetClassifier classifier;
    private BufferedImage catImage;

    @BeforeAll
    void setUp() throws IOException {
        classifier = ResNetClassifier.builder().build();
        catImage = ImageIO.read(ResNetClassifierModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @AfterAll
    void tearDown() throws Exception {
        if (classifier != null) classifier.close();
    }

    @Test
    void classify_catImage_returnsNonEmptyResults() {
        List<Classification> results = classifier.classify(catImage);

        assertFalse(results.isEmpty(), "Should return at least one classification");
        assertTrue(results.size() <= 5, "Default topK should be 5 or fewer");
    }

    @Test
    void classify_catImage_topResultHasReasonableConfidence() {
        List<Classification> results = classifier.classify(catImage);

        Classification top = results.get(0);
        assertTrue(top.confidence() > 0.1f,
                "Top result should have confidence > 0.1, got: " + top.confidence());
        assertNotNull(top.label(), "Label should not be null");
        assertFalse(top.label().isBlank(), "Label should not be blank");
    }

    @Test
    void classify_catImage_resultsAreSortedDescending() {
        List<Classification> results = classifier.classify(catImage);

        for (int i = 1; i < results.size(); i++) {
            assertTrue(results.get(i - 1).confidence() >= results.get(i).confidence(),
                    "Results should be sorted descending at index " + i);
        }
    }
}
