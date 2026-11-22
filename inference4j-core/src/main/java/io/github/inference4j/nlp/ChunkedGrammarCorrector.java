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
 * Corrects text of any length one sentence at a time, joining the corrected sentences with the original
 * whitespace.
 *
 * <pre>{@code
 * try (GrammarCorrector corrector = new ChunkedGrammarCorrector(
 *         CoeditGrammarCorrector.coeditBase().build())) {
 *     String corrected = corrector.correct(longDocument);
 * }
 * }</pre>
 *
 * <p>Grammar correction models such as CoEdIT are prompted and trained sentence by sentence, so each
 * sentence is a separate call, and text of one sentence is corrected in one call, exactly as the wrapped
 * corrector would. A sentence too long on its own is split between words. Whitespace, paragraphs and line
 * breaks are kept.
 *
 * <p>A sentence longer than a quarter of the wrapped corrector's {@code maxNewTokens} in input tokens (64
 * with the default of 256) is split, so its correction fits in the output limit; when a correction reaches
 * that limit anyway, a warning is logged.
 *
 * <p>Streaming passes the wrapped corrector's tokens through sentence by sentence, with the whitespace
 * between them; the streamed text equals {@link GenerationResult#text()}. In the result, token counts are
 * summed over all calls and the duration is the time of the whole call.
 *
 * <p>Requires a corrector that reports its token limits ({@link GrammarCorrector#correctionLimits()}),
 * such as {@link CoeditGrammarCorrector} or {@link FlanT5TextGenerator}. Thread-safe. Closing this
 * corrector closes the wrapped one.
 */
public class ChunkedGrammarCorrector implements GrammarCorrector {

    private final GrammarCorrector corrector;
    private final ChunkedGeneration chunking;

    /**
     * Wraps {@code corrector}.
     *
     * @throws UnsupportedOperationException if {@code corrector} does not report its token limits
     */
    public ChunkedGrammarCorrector(GrammarCorrector corrector) {
        this(corrector, null);
    }

    /**
     * Wraps {@code corrector}; a sentence longer than {@code maxChunkTokens} input tokens, prompt included,
     * is split between words.
     *
     * @throws IllegalArgumentException      if {@code maxChunkTokens} is less than 1
     * @throws UnsupportedOperationException if {@code corrector} does not report its token limits
     */
    public ChunkedGrammarCorrector(GrammarCorrector corrector, int maxChunkTokens) {
        this(corrector, Integer.valueOf(maxChunkTokens));
    }

    private ChunkedGrammarCorrector(GrammarCorrector corrector, Integer maxChunkTokens) {
        this.corrector = Objects.requireNonNull(corrector, "corrector");
        corrector.correctionLimits();
        this.chunking = new ChunkedGeneration(LoggerFactory.getLogger(ChunkedGrammarCorrector.class),
                maxChunkTokens, TextChunker.Packing.ONE_SENTENCE);
    }

    @Override
    public GenerationResult correct(String text, Consumer<String> tokenListener) {
        return chunking.run(text, corrector.correctionLimits(), Locale.ROOT, corrector::correct, tokenListener);
    }

    /** Closes the wrapped corrector. */
    @Override
    public void close() throws Exception {
        corrector.close();
    }
}
