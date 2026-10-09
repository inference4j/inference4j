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

import static org.assertj.core.api.Assertions.assertThat;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PunctCapSegModelModelTest {

    private PunctCapSegModel model;

    @BeforeAll
    void setUp() {
        model = PunctCapSegModel.builder().build();
    }

    @AfterAll
    void tearDown() {
        if (model != null) model.close();
    }

    @Test
    void restoresCasingPunctuationAndSentences() {
        List<String> sentences = model.infer("marie curie moved from poland to paris and later worked "
                + "with the us radium institute she won two nobel prizes");

        assertThat(sentences).hasSize(2);
        assertThat(sentences.get(0)).startsWith("Marie Curie").contains("Poland", "Paris", "U.S.").endsWith(".");
        assertThat(sentences.get(1)).startsWith("She").contains("Nobel").endsWith(".");
    }

    @Test
    void restoresAcronyms() {
        List<String> sentences = model.infer("she later worked with the us radium institute");

        assertThat(String.join(" ", sentences)).contains("U.S.");
    }

    @Test
    void reformatsUppercaseAsrOutput() {
        // Wav2Vec2 emits upper-case text without punctuation
        List<String> sentences = model.infer("MARIE CURIE MOVED TO PARIS");

        assertThat(sentences).hasSize(1);
        assertThat(sentences.get(0)).startsWith("Marie Curie").contains("Paris");
    }

    @Test
    void doesNotDoublePunctuateFormattedInput() {
        List<String> sentences = model.infer("Marie Curie moved to Paris.");

        assertThat(String.join(" ", sentences)).doesNotContain("..");
    }

    @Test
    void truncatesInputLongerThanWindowWithoutFailing() {
        String longText = "marie curie moved to paris ".repeat(100);

        List<String> sentences = model.infer(longText);

        assertThat(sentences).isNotEmpty();
        assertThat(String.join(" ", sentences).split("\\s+").length).isLessThan(500);
    }

    @Test
    void blankInputReturnsEmptyList() {
        assertThat(model.infer("   ")).isEmpty();
    }
}
