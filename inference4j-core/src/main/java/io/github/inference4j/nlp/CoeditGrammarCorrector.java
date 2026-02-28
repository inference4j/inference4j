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
 * Grammar corrector backed by CoEdit encoder-decoder models.
 *
 * <p>CoEdit (by Grammarly) is a T5-based model fine-tuned for text editing tasks.
 * This class uses the prompt prefix {@code "Fix grammatical errors in this sentence: "}
 * as documented by Grammarly for CoEdit grammar correction.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * // CoEdit Base (250M) — fast grammar correction
 * try (var corrector = CoeditGrammarCorrector.coeditBase().build()) {
 *     String fixed = corrector.correct("She don't likes the weathers today");
 *     System.out.println(fixed); // "She doesn't like the weather today"
 * }
 *
 * // CoEdit Large (780M) — best quality grammar correction
 * try (var corrector = CoeditGrammarCorrector.coeditLarge().build()) {
 *     String fixed = corrector.correct("He go to school yesterday");
 *     System.out.println(fixed); // "He went to school yesterday"
 * }
 * }</pre>
 *
 * @see TextGenerator
 * @see GrammarCorrector
 * @see GenerationResult
 */
public class CoeditGrammarCorrector implements TextGenerator, GrammarCorrector {

    private static final String GRAMMAR_PREFIX =
            "Fix grammatical errors in this sentence: ";

    private final GenerationEngine engine;

    CoeditGrammarCorrector(GenerationEngine engine) {
        this.engine = engine;
    }

    /**
     * CoEdit Base (250M parameters) preset.
     *
     * <p>Fast grammar correction. Downloads from {@code inference4j/coedit-base}
     * on first use.
     */
    public static Builder coeditBase() {
        return builder().modelId("inference4j/coedit-base");
    }

    /**
     * CoEdit Large (780M parameters) preset.
     *
     * <p>Best quality grammar correction. Downloads from {@code inference4j/coedit-large}
     * on first use. Requires external data files.
     */
    public static Builder coeditLarge() {
        return builder()
                .modelId("inference4j/coedit-large")
                .requiredFile("decoder_model.onnx_data")
                .requiredFile("encoder_model.onnx_data");
    }

    /**
     * Generic builder for custom CoEdit or compatible encoder-decoder models.
     *
     * <p>Requires at minimum a {@link Builder#modelId(String) modelId} (or
     * {@link Builder#modelSource(ModelSource) modelSource}) pointing to a directory
     * with {@code encoder_model.onnx}, {@code decoder_model.onnx},
     * {@code decoder_model_with_past.onnx}, and {@code config.json}.
     */
    public static Builder builder() {
        return new Builder();
    }

    // --- TextGenerator ---

    @Override
    public GenerationResult generate(String input) {
        return engine.generate(input);
    }

    @Override
    public GenerationResult generate(String input, Consumer<String> tokenListener) {
        return engine.generate(input, tokenListener);
    }

    // --- GrammarCorrector ---

    @Override
    public GenerationResult correct(String text, Consumer<String> tokenListener) {
        return engine.generate(GRAMMAR_PREFIX + text, tokenListener);
    }

    @Override
    public void close() throws Exception {
        engine.close();
    }

    public static class Builder
            extends AbstractEncoderDecoderBuilder<CoeditGrammarCorrector, Builder> {

        @Override
        protected TokenizerProvider defaultTokenizerProvider() {
            return SentencePieceBpeTokenizer.provider();
        }

        @Override
        protected CoeditGrammarCorrector createWrapper(GenerationEngine engine) {
            return new CoeditGrammarCorrector(engine);
        }
    }
}
