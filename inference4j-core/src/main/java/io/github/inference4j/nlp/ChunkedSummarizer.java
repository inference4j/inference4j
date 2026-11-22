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
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Summarizes text of any length chunk by chunk: cuts the text into chunks the model takes whole, summarizes
 * each, and joins the summaries as paragraphs. The summary grows with the text and covers all of it; for one
 * short summary of the whole text, use {@link MapReduceSummarizer}.
 *
 * <pre>{@code
 * try (Summarizer summarizer = new ChunkedSummarizer(BartSummarizer.distilBartCnn().build())) {
 *     String summary = summarizer.summarize(longReport);
 * }
 * }</pre>
 *
 * <p>Chunks hold whole sentences, packed across paragraphs; a sentence too long on its own is split between
 * words. By default a chunk holds up to the model's input limit. Text that fits in one call is summarized
 * exactly as the wrapped summarizer would. Each chunk summary has its leading whitespace removed, and
 * summaries are separated by a blank line.
 *
 * <p>Streaming passes each chunk's tokens through as they are generated, with the blank lines between
 * chunks; the streamed text equals {@link GenerationResult#text()}. In the result, token counts are summed
 * over all calls and the duration is the time of the whole call. When a chunk's summary reaches the
 * wrapped summarizer's {@code maxNewTokens}, a warning is logged.
 *
 * <p>Requires a summarizer that reports its token limits ({@link Summarizer#summarizationLimits()}), such as
 * {@link BartSummarizer} or {@link FlanT5TextGenerator}. Thread-safe. Closing this summarizer closes the
 * wrapped one.
 */
public class ChunkedSummarizer implements Summarizer {

    private static final Logger logger = LoggerFactory.getLogger(ChunkedSummarizer.class);
    private static final String SEPARATOR = "\n\n";

    private final Summarizer summarizer;
    private final Integer maxChunkTokens;
    private final AtomicBoolean warned = new AtomicBoolean();

    /** Wraps {@code summarizer}, with chunks of up to its input limit. */
    public ChunkedSummarizer(Summarizer summarizer) {
        this(summarizer, null);
    }

    /**
     * Wraps {@code summarizer}, with chunks of at most {@code maxChunkTokens} input tokens, prompt included,
     * or the model's input limit if lower.
     *
     * @throws IllegalArgumentException if {@code maxChunkTokens} is less than 1
     */
    public ChunkedSummarizer(Summarizer summarizer, int maxChunkTokens) {
        this(summarizer, Integer.valueOf(maxChunkTokens));
    }

    private ChunkedSummarizer(Summarizer summarizer, Integer maxChunkTokens) {
        this.summarizer = Objects.requireNonNull(summarizer, "summarizer");
        if (maxChunkTokens != null && maxChunkTokens < 1) {
            throw new IllegalArgumentException("maxChunkTokens must be at least 1, got " + maxChunkTokens);
        }
        this.maxChunkTokens = maxChunkTokens;
        summarizer.summarizationLimits();   // fail fast on a summarizer that cannot report them
    }

    @Override
    public GenerationResult summarize(String text, Consumer<String> tokenListener) {
        long start = System.nanoTime();
        TokenLimits limits = summarizer.summarizationLimits();
        int budget = maxChunkTokens != null
                ? Math.min(maxChunkTokens, limits.maxInputTokens())
                : limits.maxInputTokens();
        if (text.isBlank() || limits.countInputTokens(text) <= budget) {
            return summarizer.summarize(text, tokenListener);
        }
        TextChunker.Chunks chunks = TextChunker.split(text, Locale.ROOT,
                candidate -> limits.countInputTokens(candidate) <= budget, TextChunker.Packing.ACROSS_PARAGRAPHS);
        if (chunks.chunks().size() < 2) {
            return summarizer.summarize(text, tokenListener);
        }

        StringBuilder output = new StringBuilder();
        int promptTokens = 0;
        int generatedTokens = 0;
        for (TextChunker.Chunk chunk : chunks.chunks()) {
            ChunkListener listener = new ChunkListener(output, tokenListener);
            GenerationResult result = summarizer.summarize(chunk.text(), listener);
            promptTokens += result.promptTokens();
            generatedTokens += result.generatedTokens();
            if (limits.maxOutputTokens() > 0 && result.generatedTokens() >= limits.maxOutputTokens()) {
                warnCutOff(limits.maxOutputTokens());
            }
        }
        return new GenerationResult(output.toString(), promptTokens, generatedTokens,
                Duration.ofNanos(System.nanoTime() - start));
    }

    @Override
    public void close() throws Exception {
        summarizer.close();
    }

    private void warnCutOff(int maxOutputTokens) {
        String message = "A chunk's summary reached the limit of {} new tokens and was probably cut; "
                + "raise maxNewTokens or lower maxChunkTokens";
        if (warned.compareAndSet(false, true)) {
            logger.warn(message, maxOutputTokens);
        } else {
            logger.debug(message, maxOutputTokens);
        }
    }

    /**
     * Passes one chunk's summary on without its leading whitespace, preceded by the separator when an
     * earlier chunk produced text, and appends what it passes on to the output.
     */
    private static final class ChunkListener implements Consumer<String> {
        private final StringBuilder output;
        private final Consumer<String> listener;
        private boolean started;

        ChunkListener(StringBuilder output, Consumer<String> listener) {
            this.output = output;
            this.listener = listener;
        }

        @Override
        public void accept(String fragment) {
            String text = fragment;
            if (!started) {
                text = text.stripLeading();
                if (text.isEmpty()) {
                    return;
                }
                started = true;
                if (!output.isEmpty()) {
                    text = SEPARATOR + text;
                }
            }
            output.append(text);
            listener.accept(text);
        }
    }
}
