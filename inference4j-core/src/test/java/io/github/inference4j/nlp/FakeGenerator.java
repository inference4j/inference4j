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
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A translator and grammar corrector that upper-cases its input. One token per whitespace-separated word,
 * plus a fixed overhead per call standing in for the prompt and special tokens. Streams its output a word
 * or a whitespace run at a time, and records every input.
 */
final class FakeGenerator implements Translator, GrammarCorrector {

    private static final Pattern PIECES = Pattern.compile("\\S+|\\s+");

    final List<String> inputs = Collections.synchronizedList(new ArrayList<>());
    final int maxInputTokens;
    final int maxOutputTokens;
    final int overhead;
    /** When set, every call reports this many generated tokens, as if the output reached a limit. */
    Integer generatedTokens;
    /** Overhead of translate(text, source, target), which has a longer prompt. */
    int pairOverhead;
    boolean closed;
    GenerationResult lastResult;

    FakeGenerator(int maxInputTokens, int maxOutputTokens, int overhead) {
        this.maxInputTokens = maxInputTokens;
        this.maxOutputTokens = maxOutputTokens;
        this.overhead = overhead;
        this.pairOverhead = overhead;
    }

    static int words(String text) {
        return text.isBlank() ? 0 : text.trim().split("\\s+").length;
    }

    int count(String text) {
        return words(text) + overhead;
    }

    private GenerationResult run(String text, int callOverhead, Consumer<String> listener) {
        inputs.add(text);
        String output = text.toUpperCase(Locale.ROOT);
        Matcher pieces = PIECES.matcher(output);
        while (pieces.find()) {
            listener.accept(pieces.group());
        }
        int generated = generatedTokens != null ? generatedTokens : words(output);
        GenerationResult result = new GenerationResult(output, words(text) + callOverhead, generated, Duration.ZERO);
        lastResult = result;
        return result;
    }

    private TokenLimits limits(int callOverhead) {
        return new TokenLimits() {
            @Override
            public int maxInputTokens() {
                return maxInputTokens;
            }

            @Override
            public int maxOutputTokens() {
                return maxOutputTokens;
            }

            @Override
            public int countInputTokens(String text) {
                return words(text) + callOverhead;
            }
        };
    }

    @Override
    public GenerationResult translate(String text, Consumer<String> tokenListener) {
        return run(text, overhead, tokenListener);
    }

    @Override
    public GenerationResult translate(String text, Language source, Language target, Consumer<String> tokenListener) {
        return run(text, pairOverhead, tokenListener);
    }

    @Override
    public GenerationResult correct(String text, Consumer<String> tokenListener) {
        return run(text, overhead, tokenListener);
    }

    @Override
    public TokenLimits translationLimits() {
        return limits(overhead);
    }

    @Override
    public TokenLimits translationLimits(Language source, Language target) {
        return limits(pairOverhead);
    }

    @Override
    public TokenLimits correctionLimits() {
        return limits(overhead);
    }

    @Override
    public void close() {
        closed = true;
    }
}
