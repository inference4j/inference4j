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

import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.processing.TruncationPolicy;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.assertj.core.api.Assertions.assertThat;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ChunkedGrammarCorrectorModelTest {

    private CoeditGrammarCorrector plain;
    private ChunkedGrammarCorrector corrector;

    @BeforeAll
    void setUp() {
        plain = CoeditGrammarCorrector.coeditBase().build();
        corrector = new ChunkedGrammarCorrector(CoeditGrammarCorrector.coeditBase()
                .truncation(TruncationPolicy.FAIL)
                .build());
    }

    @AfterAll
    void tearDown() throws Exception {
        if (plain != null) plain.close();
        if (corrector != null) corrector.close();
    }

    @Test
    void correctsEachSentenceAndKeepsTheLayout() {
        String text = "She go to school yesterday. They was very happy about it.\n\nHe have two dog at home.";
        StringBuilder streamed = new StringBuilder();

        GenerationResult result = corrector.correct(text, streamed::append);

        assertThat(streamed.toString()).isEqualTo(result.text());
        String[] paragraphs = result.text().split("\n\n");
        assertThat(paragraphs).hasSize(2);
        assertThat(paragraphs[0]).isEqualTo(plain.correct("She go to school yesterday.") + " "
                + plain.correct("They was very happy about it."));
        assertThat(paragraphs[1]).isEqualTo(plain.correct("He have two dog at home."));
    }

    @Test
    void oneSentenceIsCorrectedExactlyAsByTheWrappedCorrector() {
        String sentence = "She go to school yesterday.";

        assertThat(corrector.correct(sentence)).isEqualTo(plain.correct(sentence));
    }
}
