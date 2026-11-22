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
import io.github.inference4j.processing.TokenLimits;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.IntUnaryOperator;

/**
 * A summarizer that keeps the first words of its input: {@code outputWords} maps the input's word count to
 * the summary's (it may grow, padding with "more"). One token per whitespace-separated word plus a fixed
 * overhead per call. Its output starts with a space, as DistilBART's does; it streams a word at a time and
 * records every input.
 */
final class FakeSummarizer implements Summarizer {

    final List<String> inputs = Collections.synchronizedList(new ArrayList<>());
    final int maxInputTokens;
    final int overhead;
    final IntUnaryOperator outputWords;
    boolean closed;
    GenerationResult lastResult;

    FakeSummarizer(int maxInputTokens, int overhead, IntUnaryOperator outputWords) {
        this.maxInputTokens = maxInputTokens;
        this.overhead = overhead;
        this.outputWords = outputWords;
    }

    int count(String text) {
        return FakeGenerator.words(text) + overhead;
    }

    @Override
    public GenerationResult summarize(String text, Consumer<String> tokenListener) {
        inputs.add(text);
        List<String> words = new ArrayList<>(text.isBlank() ? List.of() : Arrays.asList(text.trim().split("\\s+")));
        int keep = outputWords.applyAsInt(words.size());
        while (words.size() < keep) {
            words.add("more");
        }
        StringBuilder output = new StringBuilder();
        for (String word : words.subList(0, keep)) {
            tokenListener.accept(" " + word);
            output.append(' ').append(word);
        }
        GenerationResult result = new GenerationResult(output.toString(), count(text), keep, Duration.ZERO);
        lastResult = result;
        return result;
    }

    @Override
    public TokenLimits summarizationLimits() {
        return new TokenLimits() {
            @Override
            public int maxInputTokens() {
                return maxInputTokens;
            }

            @Override
            public int maxOutputTokens() {
                return 256;
            }

            @Override
            public int countInputTokens(String text) {
                return count(text);
            }
        };
    }

    @Override
    public void close() {
        closed = true;
    }
}
