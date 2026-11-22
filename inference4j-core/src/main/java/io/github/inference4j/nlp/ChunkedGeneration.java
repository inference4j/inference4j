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
import org.slf4j.Logger;

import java.time.Duration;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiFunction;
import java.util.function.Consumer;

/**
 * The chunk-by-chunk loop shared by {@link ChunkedTranslator} and {@link ChunkedGrammarCorrector}: cuts
 * text into chunks that fit the delegate's token budget, runs the delegate on each and joins the outputs
 * with the original whitespace. One instance per decorator; thread-safe.
 */
final class ChunkedGeneration {

    private final Logger logger;
    private final Integer maxChunkTokens;
    private final TextChunker.Packing packing;
    private final AtomicBoolean warned = new AtomicBoolean();

    /**
     * @param logger         the decorator's logger, for the cut-off warning
     * @param maxChunkTokens the chunk budget; {@code null} for a quarter of the delegate's output limit
     */
    ChunkedGeneration(Logger logger, Integer maxChunkTokens, TextChunker.Packing packing) {
        if (maxChunkTokens != null && maxChunkTokens < 1) {
            throw new IllegalArgumentException("maxChunkTokens must be at least 1, got " + maxChunkTokens);
        }
        this.logger = logger;
        this.maxChunkTokens = maxChunkTokens;
        this.packing = packing;
    }

    /**
     * Runs {@code call} on each chunk of {@code text}. Text that is a single chunk goes to {@code call}
     * unchanged, with {@code listener}, and its result is returned as is.
     */
    GenerationResult run(String text, TokenLimits limits, Locale locale,
                         BiFunction<String, Consumer<String>, GenerationResult> call, Consumer<String> listener) {
        long start = System.nanoTime();
        int budget = budget(limits);
        if (text.isBlank()
                || (packing != TextChunker.Packing.ONE_SENTENCE && limits.countInputTokens(text) <= budget)) {
            return call.apply(text, listener);
        }
        TextChunker.Chunks chunks = TextChunker.split(text, locale,
                candidate -> limits.countInputTokens(candidate) <= budget, packing);
        if (chunks.chunks().size() == 1) {
            return call.apply(text, listener);
        }

        StringBuilder output = new StringBuilder(chunks.leading());
        emit(listener, chunks.leading());
        int promptTokens = 0;
        int generatedTokens = 0;
        for (TextChunker.Chunk chunk : chunks.chunks()) {
            GenerationResult result = call.apply(chunk.text(), listener);
            output.append(result.text()).append(chunk.separator());
            emit(listener, chunk.separator());
            promptTokens += result.promptTokens();
            generatedTokens += result.generatedTokens();
            if (limits.maxOutputTokens() > 0 && result.generatedTokens() >= limits.maxOutputTokens()) {
                warnCutOff(limits.maxOutputTokens());
            }
        }
        return new GenerationResult(output.toString(), promptTokens, generatedTokens,
                Duration.ofNanos(System.nanoTime() - start));
    }

    /**
     * The most input tokens per chunk: the option, else a quarter of the output limit, never above the input
     * limit. Short chunks suit models trained on sentences: on long documents, 64-token chunks scored within
     * about a point of the best chunk size for opus-mt-en-es and 12 chrF points above 128-token chunks for
     * Flan-T5 (see LongDocumentTranslationEvaluation).
     */
    private int budget(TokenLimits limits) {
        int budget = maxChunkTokens != null ? maxChunkTokens : Math.max(1, limits.maxOutputTokens() / 4);
        return Math.min(budget, limits.maxInputTokens());
    }

    private static void emit(Consumer<String> listener, String text) {
        if (!text.isEmpty()) {
            listener.accept(text);
        }
    }

    private void warnCutOff(int maxOutputTokens) {
        String message = "A chunk's output reached the limit of {} new tokens and was probably cut; "
                + "raise maxNewTokens or lower maxChunkTokens";
        if (warned.compareAndSet(false, true)) {
            logger.warn(message, maxOutputTokens);
        } else {
            logger.debug(message, maxOutputTokens);
        }
    }
}
