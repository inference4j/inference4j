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
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class MapReduceSummarizerTest {

    /** {@code n} three-word sentences: "s1 a b. s2 a b. ..." */
    private static String sentences(int n) {
        return IntStream.rangeClosed(1, n).mapToObj(i -> "s" + i + " a b.").collect(Collectors.joining(" "));
    }

    @Test
    void textThatFitsIsSummarizedInOneCall() {
        FakeSummarizer fake = new FakeSummarizer(20, 1, n -> 2);
        MapReduceSummarizer summarizer = new MapReduceSummarizer(fake);

        GenerationResult result = summarizer.summarize(sentences(3), token -> {});

        assertThat(fake.inputs).containsExactly(sentences(3));
        assertThat(result).isSameAs(fake.lastResult);
    }

    @Test
    void longTextIsSummarizedByChunkThenReduced() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);
        MapReduceSummarizer summarizer = new MapReduceSummarizer(fake);

        GenerationResult result = summarizer.summarize(sentences(9), token -> {});

        // chunks of three sentences (9 words + 1 = 10 tokens); each summary keeps its first two words
        assertThat(fake.inputs).containsExactly(
                "s1 a b. s2 a b. s3 a b.", "s4 a b. s5 a b. s6 a b.", "s7 a b. s8 a b. s9 a b.",
                "s1 a s4 a s7 a");
        assertThat(result.text()).isEqualTo(" s1 a");
    }

    @Test
    void everyChunkFitsTheBudget() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);
        new MapReduceSummarizer(fake).summarize(sentences(40) + "\n\n" + sentences(7), token -> {});

        assertThat(fake.inputs).allSatisfy(input -> assertThat(fake.count(input)).isLessThanOrEqualTo(10));
    }

    @Test
    void onlyTheFinalCallStreams() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);
        List<String> streamed = new ArrayList<>();

        GenerationResult result = new MapReduceSummarizer(fake).summarize(sentences(9), streamed::add);

        assertThat(String.join("", streamed)).isEqualTo(result.text());
    }

    @Test
    void summariesThatStillDoNotFitAreSummarizedAgain() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> n / 2);
        MapReduceSummarizer summarizer = new MapReduceSummarizer(fake);

        summarizer.summarize(sentences(18), token -> {});

        // round 1: 6 chunks of 9 words → 6 summaries of 4 words (24); round 2 → 3 of 4 words (12);
        // round 3 → 2 summaries of 4 and 1 word (5 words fit); then the final call
        assertThat(fake.inputs).hasSize(6 + 3 + 2 + 1);
        String last = fake.inputs.get(fake.inputs.size() - 1);
        assertThat(fake.count(last)).isLessThanOrEqualTo(10);
    }

    @Test
    void tokenCountsAreSummedOverAllCalls() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);
        List<GenerationResult> results = new ArrayList<>();
        Summarizer recording = new Summarizer() {
            @Override
            public GenerationResult summarize(String text, Consumer<String> tokenListener) {
                GenerationResult result = fake.summarize(text, tokenListener);
                results.add(result);
                return result;
            }

            @Override
            public io.github.inference4j.processing.TokenLimits summarizationLimits() {
                return fake.summarizationLimits();
            }

            @Override
            public void close() {
            }
        };

        GenerationResult result = new MapReduceSummarizer(recording).summarize(sentences(9), token -> {});

        assertThat(result.promptTokens()).isEqualTo(results.stream().mapToInt(GenerationResult::promptTokens).sum());
        assertThat(result.generatedTokens())
                .isEqualTo(results.stream().mapToInt(GenerationResult::generatedTokens).sum());
        assertThat(result.duration().isNegative()).isFalse();
    }

    @Test
    void summariesThatDoNotShrinkStopTheMapStepWithAWarning() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> n + 1);
        List<GenerationResult> result = new ArrayList<>();

        List<String> warnings = WarningLog.of(MapReduceSummarizer.class,
                () -> result.add(new MapReduceSummarizer(fake).summarize(sentences(9), token -> {})));

        assertThat(warnings).singleElement().asString().contains("no shorter than their input");
        // three map calls, then the final call gets the original text
        assertThat(fake.inputs).hasSize(4);
        assertThat(fake.inputs.get(3)).isEqualTo(sentences(9));
        assertThat(result.get(0).promptTokens()).isGreaterThan(fake.count(sentences(9)));
    }

    @Test
    void theMapStepStopsAfterMaxRoundsWithAWarning() {
        // chunks of two one-word sentences (3 tokens); each summary drops a word, halving the text per round
        FakeSummarizer fake = new FakeSummarizer(3, 1, n -> n - 1);
        String text = IntStream.range(0, 1024).mapToObj(i -> "w" + i + ".").collect(Collectors.joining(" "));

        List<String> warnings = WarningLog.of(MapReduceSummarizer.class,
                () -> new MapReduceSummarizer(fake).summarize(text, token -> {}));

        assertThat(warnings).singleElement().asString().contains("after " + MapReduceSummarizer.MAX_ROUNDS + " rounds");
        assertThat(FakeGenerator.words(fake.inputs.get(fake.inputs.size() - 1))).isEqualTo(1024 >> 8);
    }

    @Test
    void aSmallerChunkBudgetMakesSmallerChunks() {
        FakeSummarizer fake = new FakeSummarizer(100, 1, n -> 2);

        new MapReduceSummarizer(fake, 7).summarize(sentences(6), token -> {});

        assertThat(fake.inputs).startsWith("s1 a b. s2 a b.", "s3 a b. s4 a b.", "s5 a b. s6 a b.");
    }

    @Test
    void anUnknownInputLimitNeverSplits() {
        FakeSummarizer fake = new FakeSummarizer(Integer.MAX_VALUE, 1, n -> 2);

        new MapReduceSummarizer(fake).summarize(sentences(500), token -> {});

        assertThat(fake.inputs).hasSize(1);
    }

    @Test
    void rejectsABudgetBelowOne() {
        assertThatThrownBy(() -> new MapReduceSummarizer(new FakeSummarizer(10, 1, n -> 2), 0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void requiresASummarizerThatReportsItsLimits() {
        Summarizer silent = new Summarizer() {
            @Override
            public GenerationResult summarize(String text, Consumer<String> tokenListener) {
                throw new AssertionError();
            }

            @Override
            public void close() {
            }
        };

        assertThatThrownBy(() -> new MapReduceSummarizer(silent))
                .isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void closingClosesTheWrappedSummarizer() throws Exception {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);
        new MapReduceSummarizer(fake).close();
        assertThat(fake.closed).isTrue();
    }
}
