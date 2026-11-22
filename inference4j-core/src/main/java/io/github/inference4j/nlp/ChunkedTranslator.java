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
import org.slf4j.LoggerFactory;

import java.util.Locale;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Translates text of any length by cutting it into chunks the model takes whole, translating each and
 * joining the translations with the original whitespace.
 *
 * <pre>{@code
 * try (Translator translator = new ChunkedTranslator(MarianTranslator.builder()
 *         .modelId("inference4j/opus-mt-en-de").build())) {
 *     String german = translator.translate(longDocument);
 * }
 * }</pre>
 *
 * <p>Chunks hold whole sentences and never span a line break, so paragraphs and line breaks are kept.
 * A sentence too long on its own is split between words. Text that fits in one chunk is translated in
 * one call, exactly as the wrapped translator would.
 *
 * <p>By default a chunk holds at most a quarter of the wrapped translator's {@code maxNewTokens} in input
 * tokens (64 with the default of 256): short chunks fit the output limit, and models trained on sentences
 * translate them best. When a chunk's translation reaches the output limit anyway, a warning is logged.
 *
 * <p>Streaming passes the wrapped translator's tokens through chunk by chunk, with the whitespace between
 * chunks; the streamed text equals {@link GenerationResult#text()}. In the result, token counts are summed
 * over all calls ({@code promptTokens} includes each call's prompt and special tokens) and the duration
 * is the time of the whole call. If a chunk fails, the exception propagates and the listener has already
 * received the chunks before it.
 *
 * <p>Requires a translator that reports its token limits ({@link Translator#translationLimits()}), such as
 * {@link MarianTranslator} or {@link FlanT5TextGenerator}. Thread-safe. Closing this translator closes the
 * wrapped one.
 */
public class ChunkedTranslator implements Translator {

    private final Translator translator;
    private final ChunkedGeneration chunking;

    /** Wraps {@code translator}, with chunks of at most a quarter of its {@code maxNewTokens}. */
    public ChunkedTranslator(Translator translator) {
        this(translator, null);
    }

    /**
     * Wraps {@code translator}, with chunks of at most {@code maxChunkTokens} input tokens, prompt
     * included, or the model's input limit if lower.
     *
     * @throws IllegalArgumentException if {@code maxChunkTokens} is less than 1
     */
    public ChunkedTranslator(Translator translator, int maxChunkTokens) {
        this(translator, Integer.valueOf(maxChunkTokens));
    }

    private ChunkedTranslator(Translator translator, Integer maxChunkTokens) {
        this.translator = Objects.requireNonNull(translator, "translator");
        this.chunking = new ChunkedGeneration(LoggerFactory.getLogger(ChunkedTranslator.class), maxChunkTokens,
                TextChunker.Packing.WITHIN_PARAGRAPH);
    }

    /**
     * {@inheritDoc}
     *
     * @throws UnsupportedOperationException if the wrapped translator requires explicit languages or does
     *                                       not report its token limits
     */
    @Override
    public GenerationResult translate(String text, Consumer<String> tokenListener) {
        return chunking.run(text, translator.translationLimits(), Locale.ROOT, translator::translate,
                tokenListener);
    }

    /**
     * {@inheritDoc}
     *
     * @throws UnsupportedOperationException if the wrapped translator has a fixed language pair or does not
     *                                       report its token limits
     */
    @Override
    public GenerationResult translate(String text, Language source, Language target,
                                      Consumer<String> tokenListener) {
        return chunking.run(text, translator.translationLimits(source, target),
                Locale.forLanguageTag(source.isoCode()),
                (chunk, listener) -> translator.translate(chunk, source, target, listener), tokenListener);
    }

    /** Closes the wrapped translator. */
    @Override
    public void close() throws Exception {
        translator.close();
    }
}
