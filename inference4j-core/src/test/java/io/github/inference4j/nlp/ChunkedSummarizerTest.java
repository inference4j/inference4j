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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ChunkedSummarizerTest {

    /** {@code n} three-word sentences: "s1 a b. s2 a b. ..." */
    private static String sentences(int n) {
        return IntStream.rangeClosed(1, n).mapToObj(i -> "s" + i + " a b.").collect(Collectors.joining(" "));
    }

    @Test
    void textThatFitsIsSummarizedInOneCall() {
        FakeSummarizer fake = new FakeSummarizer(20, 1, n -> 2);

        GenerationResult result = new ChunkedSummarizer(fake).summarize(sentences(3), token -> {});

        assertThat(fake.inputs).containsExactly(sentences(3));
        assertThat(result).isSameAs(fake.lastResult);
    }

    @Test
    void eachChunkIsSummarizedAndTheSummariesBecomeParagraphs() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);

        GenerationResult result = new ChunkedSummarizer(fake).summarize(sentences(9), token -> {});

        assertThat(fake.inputs).containsExactly(
                "s1 a b. s2 a b. s3 a b.", "s4 a b. s5 a b. s6 a b.", "s7 a b. s8 a b. s9 a b.");
        assertThat(result.text()).isEqualTo("s1 a\n\ns4 a\n\ns7 a");
    }

    @Test
    void theStreamEqualsTheText() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);
        List<String> streamed = new ArrayList<>();

        GenerationResult result = new ChunkedSummarizer(fake).summarize(sentences(9), streamed::add);

        assertThat(String.join("", streamed)).isEqualTo(result.text());
        assertThat(streamed).doesNotContain("");
    }

    @Test
    void anEmptyChunkSummaryAddsNoBlankParagraph() {
        AtomicInteger calls = new AtomicInteger();
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> calls.getAndIncrement() == 1 ? 0 : 2);

        GenerationResult result = new ChunkedSummarizer(fake).summarize(sentences(9), token -> {});

        assertThat(result.text()).isEqualTo("s1 a\n\ns7 a");
    }

    @Test
    void everyChunkFitsTheBudget() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);
        new ChunkedSummarizer(fake).summarize(sentences(40) + "\n\n" + sentences(7), token -> {});

        assertThat(fake.inputs).allSatisfy(input -> assertThat(fake.count(input)).isLessThanOrEqualTo(10));
    }

    @Test
    void aSmallerChunkBudgetMakesSmallerChunks() {
        FakeSummarizer fake = new FakeSummarizer(100, 1, n -> 2);

        new ChunkedSummarizer(fake, 7).summarize(sentences(6), token -> {});

        assertThat(fake.inputs).containsExactly("s1 a b. s2 a b.", "s3 a b. s4 a b.", "s5 a b. s6 a b.");
    }

    @Test
    void tokenCountsAreSummedOverAllCalls() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);

        GenerationResult result = new ChunkedSummarizer(fake).summarize(sentences(9), token -> {});

        assertThat(result.promptTokens()).isEqualTo(3 * 10);
        assertThat(result.generatedTokens()).isEqualTo(3 * 2);
        assertThat(result.duration().isNegative()).isFalse();
    }

    @Test
    void aSummaryCutAtTheOutputLimitIsWarnedAboutOnce() {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 300);

        List<String> warnings = WarningLog.of(ChunkedSummarizer.class,
                () -> new ChunkedSummarizer(fake).summarize(sentences(9), token -> {}));

        assertThat(warnings).singleElement().asString().contains("limit of 256 new tokens");
    }

    @Test
    void rejectsABudgetBelowOne() {
        assertThatThrownBy(() -> new ChunkedSummarizer(new FakeSummarizer(10, 1, n -> 2), 0))
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

        assertThatThrownBy(() -> new ChunkedSummarizer(silent)).isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void closingClosesTheWrappedSummarizer() throws Exception {
        FakeSummarizer fake = new FakeSummarizer(10, 1, n -> 2);
        new ChunkedSummarizer(fake).close();
        assertThat(fake.closed).isTrue();
    }
}
