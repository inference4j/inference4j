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
 * Corrects grammar and spelling errors in text.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (var corrector = CoeditGrammarCorrector.coeditBase().build()) {
 *     String fixed = corrector.correct("She don't likes the weathers today");
 *     // "She doesn't like the weather today"
 * }
 * }</pre>
 */
public interface GrammarCorrector extends AutoCloseable {

    /**
     * Corrects grammar and spelling in the given text, blocking until complete.
     *
     * @param text the text to correct
     * @return the corrected text
     */
    default String correct(String text) {
        return correct(text, token -> {}).text();
    }

    /**
     * Corrects grammar and spelling in the given text, streaming tokens.
     *
     * @param text          the text to correct
     * @param tokenListener receives each decoded text fragment as it is generated
     * @return the complete generation result
     */
    GenerationResult correct(String text, Consumer<String> tokenListener);
}
