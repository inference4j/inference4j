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

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class RougeTest {

    @Test
    void identicalTextScoresOneHundred() {
        Rouge.Scores s = Rouge.score("The cat sat on the mat.", "The cat sat on the mat.");
        assertThat(s.rouge1()).isEqualTo(100.0);
        assertThat(s.rouge2()).isEqualTo(100.0);
        assertThat(s.rougeL()).isEqualTo(100.0);
    }

    @Test
    void disjointTextScoresZero() {
        Rouge.Scores s = Rouge.score("the cat sat", "dogs run fast");
        assertThat(s.rouge1()).isZero();
        assertThat(s.rouge2()).isZero();
        assertThat(s.rougeL()).isZero();
    }

    @Test
    void scoresAKnownPair() {
        // 5 of 6 unigrams; 3 of 5 reference bigrams (4 in the hypothesis); LCS of 5
        Rouge.Scores s = Rouge.score("the cat sat on the mat", "the cat on the mat");
        assertThat(s.rouge1()).isCloseTo(100 * 2 * 1.0 * (5 / 6.0) / (1.0 + 5 / 6.0), within(1e-9));
        assertThat(s.rouge2()).isCloseTo(100 * 2 * 0.75 * 0.6 / (0.75 + 0.6), within(1e-9));
        assertThat(s.rougeL()).isCloseTo(s.rouge1(), within(1e-9));
        assertThat(s.rouge1Precision()).isEqualTo(100.0);
        assertThat(s.rouge1Recall()).isCloseTo(100 * 5 / 6.0, within(1e-9));
    }

    @Test
    void ignoresCaseAndPunctuation() {
        assertThat(Rouge.tokens("U.S. Act, (b)--Section 2!")).containsExactly("u", "s", "act", "b", "section", "2");
    }

    @Test
    void anEmptySummaryScoresZero() {
        assertThat(Rouge.score("the cat", "").rouge1()).isZero();
    }

    @Test
    void meanAveragesThePairs() {
        Rouge.Scores mean = Rouge.mean(List.of("a b", "c d"), List.of("a b", "x y"));
        assertThat(mean.rouge1()).isEqualTo(50.0);
    }
}
