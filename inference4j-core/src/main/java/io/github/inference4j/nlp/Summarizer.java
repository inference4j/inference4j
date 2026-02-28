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

import java.util.function.Consumer;

/**
 * Summarizes text into a shorter version, preserving key information.
 *
 * <p>Implementations may use encoder-decoder models like BART or Flan-T5 to produce
 * abstractive summaries. The simple {@link #summarize(String)} method blocks and returns
 * the summary text directly; the streaming overload provides token-by-token output.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (var summarizer = BartSummarizer.distilBartCnn().build()) {
 *     String summary = summarizer.summarize(longArticleText);
 * }
 * }</pre>
 */
public interface Summarizer extends AutoCloseable {

    /**
     * Summarizes the given text, blocking until the summary is complete.
     *
     * @param text the text to summarize
     * @return the summary text
     */
    default String summarize(String text) {
        return summarize(text, token -> {}).text();
    }

    /**
     * Summarizes the given text, streaming tokens to the listener as they are produced.
     *
     * @param text          the text to summarize
     * @param tokenListener receives each decoded text fragment as it is generated
     * @return the complete generation result including timing and token counts
     */
    GenerationResult summarize(String text, Consumer<String> tokenListener);
}
