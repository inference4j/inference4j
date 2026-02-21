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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MiniLMSearchRerankerModelTest {

    @Test
    void score_relevantPairHigherThanIrrelevant() {
        try (var reranker = io.github.inference4j.nlp.MiniLMSearchReranker.builder().build()) {
            float relevantScore = reranker.score("What is Java?", "Java is a programming language.");
            float irrelevantScore = reranker.score("What is Java?", "The weather is sunny today.");

            assertTrue(relevantScore > irrelevantScore,
                    "Relevant pair should score higher: relevant=" + relevantScore + " irrelevant=" + irrelevantScore);
        }
    }

    @Test
    void score_returnsValueBetweenZeroAndOne() {
        try (var reranker = MiniLMSearchReranker.builder().build()) {
            float score = reranker.score("What is Java?", "Java is a programming language.");

            assertTrue(score >= 0f && score <= 1f,
                    "Score should be between 0 and 1, got: " + score);
        }
    }

    @Test
    void scoreBatch_returnsScorePerDocument() {
        try (var reranker = MiniLMSearchReranker.builder().build()) {
            float[] scores = reranker.scoreBatch("What is Java?", java.util.List.of(
                    "Java is a programming language.",
                    "The weather is sunny today.",
                    "Java was developed by Sun Microsystems."
            ));

            assertEquals(3, scores.length, "Should return one score per document");
            for (float score : scores) {
                assertTrue(score >= 0f && score <= 1f,
                        "Each score should be between 0 and 1, got: " + score);
            }
        }
    }
}
