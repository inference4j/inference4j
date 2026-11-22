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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ChunkedTranslatorTest {

    /** Three sentences of four words each. */
    private static final String THREE_SENTENCES = "One two three four. Five six seven eight. Nine ten eleven twelve.";

    @Test
    void textThatFitsIsTranslatedInOneCall() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake)) {
            List<String> streamed = new ArrayList<>();

            GenerationResult result = translator.translate(THREE_SENTENCES, streamed::add);

            assertThat(fake.inputs).containsExactly(THREE_SENTENCES);
            assertThat(result).isSameAs(fake.lastResult);
            assertThat(String.join("", streamed)).isEqualTo(result.text());
        }
    }

    @Test
    void sentencesArePackedGreedilyWithinTheBudget() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 10)) {
            String result = translator.translate(THREE_SENTENCES);

            assertThat(fake.inputs).containsExactly(
                    "One two three four. Five six seven eight.", "Nine ten eleven twelve.");
            assertThat(fake.inputs).allSatisfy(input -> assertThat(fake.count(input)).isLessThanOrEqualTo(10));
            assertThat(result).isEqualTo(THREE_SENTENCES.toUpperCase());
        }
    }

    @Test
    void whitespaceParagraphsAndLineBreaksAreKept() throws Exception {
        String text = "  First one here. Second one here.\n\nThird para here.\n    Indented line here.\r\nLast.\n";
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 8)) {
            String result = translator.translate(text);

            assertThat(result).isEqualTo(text.toUpperCase());
            assertThat(fake.inputs).containsExactly(
                    "First one here. Second one here.", "Third para here.", "Indented line here.", "Last.");
        }
    }

    @Test
    void aSentenceTooLongOnItsOwnIsSplitBetweenWords() throws Exception {
        String text = "This sentence has far too many words to fit in one chunk, so it is cut between words. Short.";
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 6)) {
            String result = translator.translate(text);

            assertThat(result).isEqualTo(text.toUpperCase());
            assertThat(fake.inputs).hasSizeGreaterThan(3);
            assertThat(fake.inputs).allSatisfy(input -> assertThat(fake.count(input)).isLessThanOrEqualTo(6));
            // The rest of the long sentence and the next sentence share a chunk
            assertThat(fake.inputs).last().isEqualTo("cut between words. Short.");
        }
    }

    @Test
    void aWordTooLongOnItsOwnIsPassedWhole() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 3);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 2)) {
            String result = translator.translate("Alpha beta. Gamma.");

            assertThat(result).isEqualTo("ALPHA BETA. GAMMA.");
            assertThat(fake.inputs).containsExactly("Alpha", "beta", ".", "Gamma", ".");
        }
    }

    @Test
    void streamedTextEqualsTheResultText() throws Exception {
        String text = "\n First one here. Second one here.\n\nThird para here.  ";
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 5)) {
            StringBuilder streamed = new StringBuilder();

            GenerationResult result = translator.translate(text, streamed::append);

            assertThat(streamed.toString()).isEqualTo(result.text()).isEqualTo(text.toUpperCase());
        }
    }

    @Test
    void tokenCountsAreSummedOverTheChunks() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 10)) {
            GenerationResult result = translator.translate(THREE_SENTENCES, token -> {});

            assertThat(result.promptTokens()).isEqualTo(9 + 5);
            assertThat(result.generatedTokens()).isEqualTo(12);
            assertThat(result.duration().isNegative()).isFalse();
        }
    }

    @Test
    void aLanguagePairUsesItsOwnLimits() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        fake.pairOverhead = 5;
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 10)) {
            String result = translator.translate(THREE_SENTENCES, Language.EN, Language.DE);

            assertThat(fake.inputs).containsExactly(
                    "One two three four.", "Five six seven eight.", "Nine ten eleven twelve.");
            assertThat(result).isEqualTo(THREE_SENTENCES.toUpperCase());
        }
    }

    @Test
    void aTranslatorWithoutLimitsIsReportedWhenUsed() throws Exception {
        Translator plain = new Translator() {
            @Override
            public GenerationResult translate(String text, Consumer<String> tokenListener) {
                throw new AssertionError("not called");
            }

            @Override
            public GenerationResult translate(String text, Language source, Language target,
                                              Consumer<String> tokenListener) {
                throw new AssertionError("not called");
            }

            @Override
            public void close() {
            }
        };
        try (ChunkedTranslator translator = new ChunkedTranslator(plain)) {
            assertThatThrownBy(() -> translator.translate("Hello."))
                    .isInstanceOf(UnsupportedOperationException.class)
                    .hasMessageContaining("does not report its token limits");
        }
    }

    @Test
    void maxChunkTokensMustBePositive() {
        assertThatThrownBy(() -> new ChunkedTranslator(new FakeGenerator(512, 200, 1), 0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void theDefaultBudgetIsAQuarterOfTheOutputLimit() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 40, 1);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake)) {
            translator.translate(THREE_SENTENCES);

            assertThat(fake.inputs).containsExactly(
                    "One two three four. Five six seven eight.", "Nine ten eleven twelve.");
        }
    }

    @Test
    void theBudgetNeverExceedsTheInputLimit() throws Exception {
        FakeGenerator fake = new FakeGenerator(6, 200, 1);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 100)) {
            translator.translate(THREE_SENTENCES);

            assertThat(fake.inputs).hasSize(3);
        }
    }

    @Test
    void outputThatReachesTheLimitIsWarnedOnce() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        fake.generatedTokens = 200;
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 5)) {
            List<String> warnings = WarningLog.of(ChunkedTranslator.class, () -> {
                translator.translate(THREE_SENTENCES);
                translator.translate(THREE_SENTENCES);
            });

            assertThat(warnings).hasSize(1);
            assertThat(warnings.get(0)).contains("200 new tokens");
        }
    }

    @Test
    void closingClosesTheWrappedTranslator() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        new ChunkedTranslator(fake).close();

        assertThat(fake.closed).isTrue();
    }

    @Test
    void blankTextGoesStraightThrough() throws Exception {
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 1)) {
            translator.translate("  ");

            assertThat(fake.inputs).containsExactly("  ");
        }
    }

    @Test
    void concurrentCallsGiveTheSameResults() throws Exception {
        String text = THREE_SENTENCES + "\n\n" + THREE_SENTENCES;
        FakeGenerator fake = new FakeGenerator(512, 200, 1);
        ExecutorService executor = Executors.newFixedThreadPool(4);
        try (ChunkedTranslator translator = new ChunkedTranslator(fake, 7)) {
            String expected = translator.translate(text);
            List<Future<String>> results = new ArrayList<>();
            for (int i = 0; i < 80; i++) {
                results.add(executor.submit(() -> translator.translate(text)));
            }
            for (Future<String> result : results) {
                assertThat(result.get(10, TimeUnit.SECONDS)).isEqualTo(expected);
            }
        } finally {
            executor.shutdownNow();
        }
    }
}
