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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

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

    private static final String LONG_TEXT = "marie curie moved to paris ".repeat(100);

    @Test
    void failPolicyRejectsInputOverTheTokenLimit() {
        try (var strict = PunctCapSegModel.builder()
                .truncation(TruncationPolicy.FAIL)
                .build()) {
            assertThatThrownBy(() -> strict.infer(LONG_TEXT))
                    .isInstanceOf(InputTooLongException.class);
            assertThat(strict.infer("marie curie moved to paris")).isNotEmpty();
        }
    }

    // Well over the 254-piece window
    private static final String LONG_TRANSCRIPT = "marie curie moved to paris she won two nobel prizes ".repeat(60);

    private static List<String> words(List<String> sentences) {
        return java.util.Arrays.stream(String.join(" ", sentences).toLowerCase()
                        .replaceAll("[^a-z' ]", " ").trim().split("\\s+"))
                .toList();
    }

    @Test
    void strideFormatsInputBeyondTheWindowInFull() {
        try (var windowed = PunctCapSegModel.builder().stride(64).build()) {
            List<String> sentences = windowed.infer(LONG_TRANSCRIPT);

            assertThat(words(sentences)).isEqualTo(List.of(LONG_TRANSCRIPT.trim().split("\\s+")));
            assertThat(sentences).allSatisfy(s -> assertThat(s).isNotBlank());
            assertThat(String.join(" ", sentences)).contains("Marie Curie", "Paris", "Nobel");
        }
    }

    @Test
    void withoutStrideInputBeyondTheWindowIsCut() {
        List<String> sentences = model.infer(LONG_TRANSCRIPT);

        assertThat(words(sentences).size()).isLessThan(LONG_TRANSCRIPT.trim().split("\\s+").length);
    }
}
