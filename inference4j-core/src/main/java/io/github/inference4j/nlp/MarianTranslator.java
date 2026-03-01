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

import io.github.inference4j.generation.GenerationEngine;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.tokenizer.SentencePieceBpeTokenizer;
import io.github.inference4j.tokenizer.TokenizerProvider;

import java.util.function.Consumer;

/**
 * Translator backed by MarianMT encoder-decoder models.
 *
 * <p>MarianMT models are language-pair-specific: each model handles exactly one
 * source-target pair (e.g., {@code Helsinki-NLP/opus-mt-en-de} translates English
 * to German). The language pair is baked into the model, so input text is passed
 * directly to the encoder without any task prefix.
 *
 * <p>Because the pair is fixed, {@link #translate(String)} works out of the box,
 * while {@link #translate(String, Language, Language)} throws
 * {@link UnsupportedOperationException}.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // English to German
 * try (var translator = MarianTranslator.builder()
 *         .modelId("Helsinki-NLP/opus-mt-en-de").build()) {
 *     String german = translator.translate("Hello, how are you?");
 *     System.out.println(german);
 * }
 *
 * // English to French with streaming
 * try (var translator = MarianTranslator.builder()
 *         .modelId("Helsinki-NLP/opus-mt-en-fr").build()) {
 *     translator.translate("Good morning!", token -> System.out.print(token));
 * }
 * }</pre>
 *
 * @see TextGenerator
 * @see Translator
 * @see GenerationResult
 */
public class MarianTranslator implements TextGenerator, Translator {

    private final GenerationEngine engine;

    MarianTranslator(GenerationEngine engine) {
        this.engine = engine;
    }

    /**
     * Generic builder for MarianMT or compatible encoder-decoder translation models.
     *
     * <p>Requires a {@link Builder#modelId(String) modelId} (or
     * {@link Builder#modelSource(ModelSource) modelSource}) pointing to a directory
     * with {@code encoder_model.onnx}, {@code decoder_model.onnx},
     * {@code decoder_with_past_model.onnx}, and {@code config.json}.
     */
    public static Builder builder() {
        return new Builder();
    }


    @Override
    public GenerationResult generate(String input) {
        return engine.generate(input);
    }

    @Override
    public GenerationResult generate(String input, Consumer<String> tokenListener) {
        return engine.generate(input, tokenListener);
    }

    // --- Translator (baked-in pair — text passed directly) ---

    @Override
    public GenerationResult translate(String text, Consumer<String> tokenListener) {
        return engine.generate(text, tokenListener);
    }

    @Override
    public GenerationResult translate(String text, Language source, Language target,
                                       Consumer<String> tokenListener) {
        throw new UnsupportedOperationException(
                "MarianMT models have a fixed language pair. Use translate(text) instead.");
    }

    @Override
    public void close() throws Exception {
        engine.close();
    }

    public static class Builder
            extends AbstractEncoderDecoderBuilder<MarianTranslator, Builder> {

        @Override
        protected TokenizerProvider defaultTokenizerProvider() {
            return SentencePieceBpeTokenizer.provider();
        }

        @Override
        protected MarianTranslator createWrapper(GenerationEngine engine) {
            return new MarianTranslator(engine);
        }
    }
}
