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

import io.github.inference4j.exception.InputTooLongException;
import io.github.inference4j.processing.TruncationPolicy;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

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

        assertThat(relevantScore > irrelevantScore).as("Relevant pair should score higher: relevant=" + relevantScore + " irrelevant=" + irrelevantScore).isTrue();
    }

    @Test
    void score_returnsValueBetweenZeroAndOne() {
        float score = reranker.score("What is Java?", "Java is a programming language.");

        assertThat(score >= 0f && score <= 1f).as("Score should be between 0 and 1, got: " + score).isTrue();
    }

    @Test
    void scoreBatch_returnsScorePerDocument() {
        float[] scores = reranker.scoreBatch("What is Java?", List.of(
                "Java is a programming language.",
                "The weather is sunny today.",
                "Java was developed by Sun Microsystems."
        ));

        assertThat(scores.length).as("Should return one score per document").isEqualTo(3);
        for (float score : scores) {
            assertThat(score >= 0f && score <= 1f).as("Each score should be between 0 and 1, got: " + score).isTrue();
        }
    }

    private static final String LONG_TEXT = "java is a programming language ".repeat(200);

    @Test
    void failPolicyRejectsInputOverTheTokenLimit() {
        try (var strict = MiniLMSearchReranker.builder()
                .truncation(TruncationPolicy.FAIL)
                .build()) {
            assertThatThrownBy(() -> strict.score("what is java", LONG_TEXT))
                    .isInstanceOf(InputTooLongException.class);
            assertThat(strict.score("what is java", "Java is a programming language.")).isBetween(0f, 1f);
        }
    }

    private static final String QUERY = "How many people live in Berlin?";
    // ~600 tokens of unrelated text, then the answer at the very end
    private static final String LONG_DOCUMENT_ANSWER_AT_END =
            "The orchestra rehearsed the symphony for weeks before the season opened in the autumn. ".repeat(40)
                    + "Berlin has a population of about 3.7 million people, making it the largest city in Germany.";
    private static final String LONG_DOCUMENT_IRRELEVANT =
            "The orchestra rehearsed the symphony for weeks before the season opened in the autumn. ".repeat(41);

    @Test
    void strideScoresARelevantPassageBeyondTheTokenLimit() {
        // ms-marco-MiniLM was trained on short passages; see the reranker docs for passage size
        try (var maxP = MiniLMSearchReranker.builder().maxLength(128).stride(32).build()) {
            float truncatedScore = reranker.score(QUERY, LONG_DOCUMENT_ANSWER_AT_END);
            float maxPScore = maxP.score(QUERY, LONG_DOCUMENT_ANSWER_AT_END);

            assertThat(truncatedScore).as("truncated: the answer is cut off").isLessThan(0.1f);
            assertThat(maxPScore).as("MaxP: the answer passage is scored").isGreaterThan(0.5f);
        }
    }

    @Test
    void strideRanksTheDocumentWithTheAnswerFirst() {
        try (var maxP = MiniLMSearchReranker.builder().maxLength(128).stride(32).build()) {
            float[] scores = maxP.scoreBatch(QUERY, List.of(LONG_DOCUMENT_IRRELEVANT, LONG_DOCUMENT_ANSWER_AT_END));

            assertThat(scores[1]).isGreaterThan(scores[0]);
        }
    }

    @Test
    void strideGivesSameScoreForShortDocuments() {
        String document = "Berlin has a population of about 3.7 million people.";
        try (var maxP = MiniLMSearchReranker.builder().stride(64).build()) {
            assertThat(maxP.score(QUERY, document)).isEqualTo(reranker.score(QUERY, document));
        }
    }
}
