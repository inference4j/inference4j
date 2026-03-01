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

import static org.assertj.core.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class EfficientNetClassifierModelTest {

    private EfficientNetClassifier classifier;
    private BufferedImage catImage;

    @BeforeAll
    void setUp() throws IOException {
        classifier = EfficientNetClassifier.builder().build();
        catImage = ImageIO.read(EfficientNetClassifierModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @AfterAll
    void tearDown() throws Exception {
        if (classifier != null) classifier.close();
    }

    @Test
    void classify_catImage_returnsNonEmptyResults() {
        List<Classification> results = classifier.classify(catImage);

        assertThat(results.isEmpty()).as("Should return at least one classification").isFalse();
        assertThat(results.size() <= 5).as("Default topK should be 5 or fewer").isTrue();
    }

    @Test
    void classify_catImage_topResultHasReasonableConfidence() {
        List<Classification> results = classifier.classify(catImage);

        Classification top = results.get(0);
        assertThat(top.confidence() > 0.1f).as("Top result should have confidence > 0.1, got: " + top.confidence()).isTrue();
        assertThat(top.label()).as("Label should not be null").isNotNull();
        assertThat(top.label().isBlank()).as("Label should not be blank").isFalse();
    }

    @Test
    void classify_catImage_resultsAreSortedDescending() {
        List<Classification> results = classifier.classify(catImage);

        for (int i = 1; i < results.size(); i++) {
            assertThat(results.get(i - 1).confidence() >= results.get(i).confidence()).as("Results should be sorted descending at index " + i).isTrue();
        }
    }
}
