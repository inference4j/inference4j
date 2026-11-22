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

class ChrFTest {

    @Test
    void identicalTextScoresOneHundred() {
        assertThat(ChrF.of("Der Hund schläft.", "Der Hund schläft.")).isCloseTo(100, within(1e-9));
    }

    @Test
    void whitespaceIsIgnored() {
        assertThat(ChrF.of("Der Hund schläft.", "Der  Hund\nschläft .")).isCloseTo(100, within(1e-9));
    }

    @Test
    void textWithNoCharacterInCommonScoresZero() {
        assertThat(ChrF.of("abc", "xyz")).isZero();
        assertThat(ChrF.of("abc", "")).isZero();
    }

    @Test
    void aCloserHypothesisScoresHigher() {
        String reference = "Die Katze sitzt auf der Matte.";
        double close = ChrF.of(reference, "Die Katze sitzt auf einer Matte.");
        double far = ChrF.of(reference, "Ein Hund liegt im Garten.");

        assertThat(close).isBetween(50.0, 100.0);
        assertThat(far).isLessThan(close);
    }

    @Test
    void recallWeighsMoreThanPrecision() {
        String reference = "abcdefgh";
        double missingHalf = ChrF.of(reference, "abcd");
        double extraHalf = ChrF.of("abcd", reference);

        assertThat(missingHalf).isLessThan(extraHalf);
    }

    @Test
    void corpusScoreSumsStatisticsOverDocuments() {
        double corpus = ChrF.corpus(List.of("abcdef", "ghijkl"), List.of("abcdef", "xyzuvw"));

        assertThat(corpus).isBetween(ChrF.of("ghijkl", "xyzuvw"), ChrF.of("abcdef", "abcdef"));
    }
}
