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

import io.github.inference4j.ZeroShotInput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.data.Offset.offset;

/**
 * Expected scores are HuggingFace's {@code zero-shot-classification} pipeline (PyTorch weights, template
 * "This text is about {}."). The ONNX export differs from PyTorch by up to ~1e-4 on texts this short.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class DebertaZeroShotClassifierModelTest {

    private static final List<String> TICKET_LABELS = List.of("billing", "outage", "feature request", "account access");

    private DebertaZeroShotClassifier classifier;

    @BeforeAll
    void setUp() {
        classifier = DebertaZeroShotClassifier.builder().build();
    }

    @AfterAll
    void tearDown() {
        if (classifier != null) {
            classifier.close();
        }
    }

    @Test
    void picksTheLabelEachTicketIsAbout() {
        assertThat(classifier.classify("My invoice shows the same charge twice.", TICKET_LABELS).get(0).label())
                .isEqualTo("billing");
        assertThat(classifier.classify("The service has been unavailable since this morning and nothing loads.",
                TICKET_LABELS).get(0).label()).isEqualTo("outage");
        assertThat(classifier.classify("Please add a dark mode to the app.", TICKET_LABELS).get(0).label())
                .isEqualTo("feature request");
    }

    @Test
    void singleLabelScoresMatchTheReferenceAndSumToOne() {
        List<TextClassification> result = classifier.classify("My invoice shows the same charge twice.", TICKET_LABELS);

        assertThat(result).extracting(TextClassification::label)
                .containsExactly("billing", "account access", "outage", "feature request");
        assertThat(result.get(0).confidence()).isCloseTo(0.9656f, offset(1e-3f));
        assertThat(result.get(1).confidence()).isCloseTo(0.0249f, offset(1e-3f));
        assertThat(result.get(0).index()).isEqualTo(0);   // position in the list passed in
        assertThat(result.stream().mapToDouble(TextClassification::confidence).sum()).isCloseTo(1.0, offset(1e-4));
    }

    @Test
    void multiLabelScoresEachLabelOnItsOwn() {
        try (var multi = DebertaZeroShotClassifier.builder().multiLabel(true).build()) {
            List<TextClassification> result = multi.classify(
                    "I can't log in after resetting my password, and I was charged for a plan I cancelled.",
                    List.of("billing", "account access", "outage"));

            assertThat(result.get(0).label()).isEqualTo("account access");
            assertThat(result.get(0).confidence()).isCloseTo(0.9698f, offset(2e-3f));
            assertThat(result.get(1).label()).isEqualTo("billing");
            assertThat(result.get(1).confidence()).isCloseTo(0.2713f, offset(2e-3f));
            // Not a distribution: the scores need not sum to 1
            assertThat(result.stream().mapToDouble(TextClassification::confidence).sum()).isGreaterThan(1.2);
        }
    }

    @Test
    void aSingleCandidateLabelIsScoredOnItsOwn() {
        List<TextClassification> result = classifier.classify("Apple unveiled a new chip that doubles battery life.",
                List.of("technology"));

        assertThat(result).singleElement().satisfies(r -> assertThat(r.confidence()).isCloseTo(0.9875f, offset(1e-3f)));
    }

    @Test
    void longTextIsTruncatedOrScoredInWindows() {
        String article = String.join(" ", Collections.nCopies(3,
                "Our company reported strong quarterly earnings, beating analyst expectations on revenue and profit."))
                + " " + String.join(" ", Collections.nCopies(40,
                "The central bank kept interest rates unchanged while inflation stayed close to target."));
        List<String> labels = List.of("economy", "sports", "technology", "health");

        assertThat(classifier.classify(article, labels).get(0).label()).isEqualTo("economy");
        try (var windowed = DebertaZeroShotClassifier.builder().stride(64).build()) {
            assertThat(windowed.classify(article, labels).get(0).label()).isEqualTo("economy");
        }
    }

    @Test
    void worksThroughTheZeroShotClassifierInterface() {
        ZeroShotTextClassifier zeroShot = classifier;

        List<TextClassification> result = zeroShot.run(new ZeroShotInput<>("Please add a dark mode to the app.", TICKET_LABELS));

        assertThat(result.get(0).label()).isEqualTo("feature request");
    }

    @Test
    void largeModelMatchesTheReference() {
        try (var large = DebertaZeroShotClassifier.large().build()) {
            List<TextClassification> result = large.classify("My invoice shows the same charge twice.", TICKET_LABELS);

            assertThat(result.get(0).label()).isEqualTo("billing");
            assertThat(result.get(0).confidence()).isCloseTo(0.9973f, offset(1e-3f));
        }
    }

    @Test
    void rejectsATemplateWithoutAPlaceholder() {
        assertThatThrownBy(() -> DebertaZeroShotClassifier.builder().hypothesisTemplate("This text is about it.").build())
                .isInstanceOf(IllegalArgumentException.class);
    }
}
