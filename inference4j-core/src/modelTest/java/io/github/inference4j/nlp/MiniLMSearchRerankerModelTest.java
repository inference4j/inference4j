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
class MiniLMSearchRerankerModelTest {

    private MiniLMSearchReranker reranker;

    @BeforeAll
    void setUp() {
        reranker = MiniLMSearchReranker.builder().build();
    }

    @AfterAll
    void tearDown() throws Exception {
        if (reranker != null) reranker.close();
    }

    @Test
    void score_relevantPairHigherThanIrrelevant() {
        float relevantScore = reranker.score("What is Java?", "Java is a programming language.");
        float irrelevantScore = reranker.score("What is Java?", "The weather is sunny today.");

        assertTrue(relevantScore > irrelevantScore,
                "Relevant pair should score higher: relevant=" + relevantScore + " irrelevant=" + irrelevantScore);
    }

    @Test
    void score_returnsValueBetweenZeroAndOne() {
        float score = reranker.score("What is Java?", "Java is a programming language.");

        assertTrue(score >= 0f && score <= 1f,
                "Score should be between 0 and 1, got: " + score);
    }

    @Test
    void scoreBatch_returnsScorePerDocument() {
        float[] scores = reranker.scoreBatch("What is Java?", List.of(
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
