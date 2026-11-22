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
import org.junit.jupiter.api.Test;

import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ChunkedGrammarCorrectorTest {

    @Test
    void oneSentenceIsCorrectedInOneCall() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedGrammarCorrector corrector = new ChunkedGrammarCorrector(fake)) {
            GenerationResult result = corrector.correct("She go to school yesterday.", token -> {});

            assertThat(fake.inputs).containsExactly("She go to school yesterday.");
            assertThat(result).isSameAs(fake.lastResult);
        }
    }

    @Test
    void eachSentenceIsCorrectedSeparately() throws Exception {
        String text = "She go home. They was late.\n\n  He have a dog.\n";
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedGrammarCorrector corrector = new ChunkedGrammarCorrector(fake)) {
            StringBuilder streamed = new StringBuilder();

            GenerationResult result = corrector.correct(text, streamed::append);

            assertThat(fake.inputs).containsExactly("She go home.", "They was late.", "He have a dog.");
            assertThat(result.text()).isEqualTo(text.toUpperCase());
            assertThat(streamed.toString()).isEqualTo(result.text());
        }
    }

    @Test
    void aSentenceTooLongOnItsOwnIsSplitBetweenWords() throws Exception {
        String text = "This sentence goes on and on without end and it never seems to stop at all. Ok.";
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedGrammarCorrector corrector = new ChunkedGrammarCorrector(fake, 6)) {
            String result = corrector.correct(text);

            assertThat(result).isEqualTo(text.toUpperCase());
            assertThat(fake.inputs).allSatisfy(input -> assertThat(fake.count(input)).isLessThanOrEqualTo(6));
            assertThat(fake.inputs).last().isEqualTo("Ok.");
        }
    }

    @Test
    void aCorrectorWithoutLimitsIsRejected() {
        GrammarCorrector plain = new GrammarCorrector() {
            @Override
            public GenerationResult correct(String text, Consumer<String> tokenListener) {
                throw new AssertionError("not called");
            }

            @Override
            public void close() {
            }
        };

        assertThatThrownBy(() -> new ChunkedGrammarCorrector(plain))
                .isInstanceOf(UnsupportedOperationException.class)
                .hasMessageContaining("does not report its token limits");
    }

    @Test
    void closingClosesTheWrappedCorrector() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        new ChunkedGrammarCorrector(fake).close();

        assertThat(fake.closed).isTrue();
    }
}
