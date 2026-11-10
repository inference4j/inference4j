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

package io.github.inference4j.audio;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class WordErrorRateTest {

    @Test
    void identicalTextIsZero() {
        assertThat(WordErrorRate.of("ask not what", "ask not what")).isZero();
    }

    @Test
    void ignoresCaseAndPunctuation() {
        assertThat(WordErrorRate.of("And so, my fellow Americans.", "and so my fellow americans")).isZero();
    }

    @Test
    void countsSubstitutionsInsertionsAndDeletions() {
        // substitution (ask -> as), deletion (not), insertion (really): 3 errors over 4 reference words
        assertThat(WordErrorRate.of("ask not what you", "as what really you")).isCloseTo(0.75, within(1e-9));
    }

    @Test
    void emptyHypothesisIsAllErrors() {
        assertThat(WordErrorRate.of("ask not what", "")).isEqualTo(1.0);
    }

    @Test
    void emptyReferenceAndHypothesisIsZero() {
        assertThat(WordErrorRate.of("", "  ")).isZero();
    }
}
