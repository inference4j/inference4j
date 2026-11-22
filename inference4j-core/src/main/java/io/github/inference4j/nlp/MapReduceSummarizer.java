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
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Summarizes text of any length by map-reduce: cuts the text into chunks the model takes whole, summarizes
 * each (map), joins the summaries and summarizes the result (reduce). When the joined summaries are still
 * too long, the map step runs again on them.
 *
 * <pre>{@code
 * try (Summarizer summarizer = new MapReduceSummarizer(BartSummarizer.distilBartCnn().build())) {
 *     String summary = summarizer.summarize(longReport);
 * }
 * }</pre>
 *
 * <p>Chunks hold whole sentences, packed across paragraphs; a sentence too long on its own is split between
 * words. By default a chunk holds up to the model's input limit. Text that fits in one call is summarized
 * exactly as the wrapped summarizer would.
 *
 * <p>Only the final call streams to the listener; the chunk summaries are not streamed. In the result, token
 * counts are summed over all calls and the duration is the time of the whole call. Each chunk is summarized
 * without the rest of the document, so the summary can miss links between distant parts.
 *
 * <p>Requires a summarizer that reports its token limits ({@link Summarizer#summarizationLimits()}), such as
 * {@link BartSummarizer} or {@link FlanT5TextGenerator}. Thread-safe. Closing this summarizer closes the
 * wrapped one.
 */
public class MapReduceSummarizer implements Summarizer {

    /** The most map rounds before the reduce call runs on whatever is left. */
    static final int MAX_ROUNDS = 8;

    private static final Logger logger = LoggerFactory.getLogger(MapReduceSummarizer.class);

    private final Summarizer summarizer;
    private final Integer maxChunkTokens;
    private final AtomicBoolean warned = new AtomicBoolean();

    /** Wraps {@code summarizer}, with chunks of up to its input limit. */
    public MapReduceSummarizer(Summarizer summarizer) {
        this(summarizer, null);
    }

    /**
     * Wraps {@code summarizer}, with chunks of at most {@code maxChunkTokens} input tokens, prompt included,
     * or the model's input limit if lower.
     *
     * @throws IllegalArgumentException if {@code maxChunkTokens} is less than 1
     */
    public MapReduceSummarizer(Summarizer summarizer, int maxChunkTokens) {
        this(summarizer, Integer.valueOf(maxChunkTokens));
    }

    private MapReduceSummarizer(Summarizer summarizer, Integer maxChunkTokens) {
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

        String current = text;
        int promptTokens = 0;
        int generatedTokens = 0;
        int currentTokens = text.isBlank() ? 0 : limits.countInputTokens(text);
        boolean mapped = false;
        for (int round = 0; currentTokens > budget; round++) {
            if (round == MAX_ROUNDS) {
                warn("The summaries still exceed {} tokens after {} rounds; the final call gets them as they are",
                        budget, MAX_ROUNDS);
                break;
            }
            TextChunker.Chunks chunks = TextChunker.split(current, Locale.ROOT,
                    candidate -> limits.countInputTokens(candidate) <= budget, TextChunker.Packing.ACROSS_PARAGRAPHS);
            if (chunks.chunks().size() < 2) {
                break;
            }
            mapped = true;
            List<String> summaries = new ArrayList<>();
            for (TextChunker.Chunk chunk : chunks.chunks()) {
                GenerationResult result = summarizer.summarize(chunk.text(), token -> {});
                promptTokens += result.promptTokens();
                generatedTokens += result.generatedTokens();
                String summary = result.text().strip();
                if (!summary.isEmpty()) {
                    summaries.add(summary);
                }
            }
            String joined = String.join(" ", summaries);
            int joinedTokens = limits.countInputTokens(joined);
            if (joinedTokens >= currentTokens) {
                warn("The chunk summaries ({} tokens) are no shorter than their input ({} tokens); "
                        + "the final call gets the input as it is", joinedTokens, currentTokens);
                break;
            }
            current = joined;
            currentTokens = joinedTokens;
        }

        GenerationResult result = summarizer.summarize(current, tokenListener);
        if (!mapped) {
            return result;
        }
        return new GenerationResult(result.text(), promptTokens + result.promptTokens(),
                generatedTokens + result.generatedTokens(), Duration.ofNanos(System.nanoTime() - start));
    }

    @Override
    public void close() throws Exception {
        summarizer.close();
    }

    private void warn(String message, Object first, Object second) {
        if (warned.compareAndSet(false, true)) {
            logger.warn(message, first, second);
        } else {
            logger.debug(message, first, second);
        }
    }
}
