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
 * Translates text between languages.
 *
 * <p>Two usage modes are supported:
 * <ul>
 *   <li><strong>Baked-in pair:</strong> Models like MarianMT are trained on a specific
 *       language pair. Use {@link #translate(String)} — the pair is determined by the model.</li>
 *   <li><strong>Specified pair:</strong> Versatile models like Flan-T5 can translate between
 *       any supported pair. Use {@link #translate(String, Language, Language)}.</li>
 * </ul>
 *
 * <p>Implementations should throw {@link UnsupportedOperationException} for the mode
 * they do not support.
 *
 * <p>Example usage:
 * <pre>{@code
 * // MarianMT — pair is baked into the model
 * try (var translator = MarianTranslator.builder()
 *         .modelId("Helsinki-NLP/opus-mt-en-de").build()) {
 *     String german = translator.translate("Hello, how are you?");
 * }
 *
 * // Flan-T5 — specify the pair
 * try (var translator = FlanT5TextGenerator.flanT5Base().build()) {
 *     String french = translator.translate("Hello world", Language.EN, Language.FR);
 * }
 * }</pre>
 */
public interface Translator extends AutoCloseable {

    /**
     * Translates text using the model's baked-in language pair.
     *
     * @param text the text to translate
     * @return the translated text
     * @throws UnsupportedOperationException if the model requires explicit language specification
     */
    default String translate(String text) {
        return translate(text, token -> {}).text();
    }

    /**
     * Translates text using the model's baked-in language pair, streaming tokens.
     *
     * @param text          the text to translate
     * @param tokenListener receives each decoded text fragment as it is generated
     * @return the complete generation result
     * @throws UnsupportedOperationException if the model requires explicit language specification
     */
    GenerationResult translate(String text, Consumer<String> tokenListener);

    /**
     * Translates text between the specified languages.
     *
     * @param text   the text to translate
     * @param source the source language
     * @param target the target language
     * @return the translated text
     * @throws UnsupportedOperationException if the model only supports a fixed language pair
     */
    default String translate(String text, Language source, Language target) {
        return translate(text, source, target, token -> {}).text();
    }

    /**
     * Translates text between the specified languages, streaming tokens.
     *
     * @param text          the text to translate
     * @param source        the source language
     * @param target        the target language
     * @param tokenListener receives each decoded text fragment as it is generated
     * @return the complete generation result
     * @throws UnsupportedOperationException if the model only supports a fixed language pair
     */
    GenerationResult translate(String text, Language source, Language target, Consumer<String> tokenListener);
}
