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
import io.github.inference4j.tokenizer.TokenizerProvider;
import io.github.inference4j.tokenizer.UnigramTokenizer;

import java.util.function.Consumer;

/**
 * Text generator backed by Flan-T5 encoder-decoder models.
 *
 * <p>Flan-T5 is a versatile instruction-tuned model that handles multiple NLP tasks
 * via prompt prefixes. This class implements {@link TextGenerator}, {@link Summarizer},
 * {@link Translator}, {@link SqlGenerator}, and {@link GrammarCorrector}, each
 * prepending the appropriate task prefix before delegating to the underlying
 * {@link GenerationEngine}.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * // Flan-T5 Small (77M) — fast, lightweight
 * try (var gen = FlanT5TextGenerator.flanT5Small().build()) {
 *     System.out.println(gen.summarize("Long article text..."));
 * }
 *
 * // Flan-T5 Base (250M) — good balance of speed and quality
 * try (var gen = FlanT5TextGenerator.flanT5Base().build()) {
 *     String french = gen.translate("Hello world", Language.EN, Language.FR);
 *     System.out.println(french);
 * }
 *
 * // Flan-T5 Large (780M) — best quality
 * try (var gen = FlanT5TextGenerator.flanT5Large().build()) {
 *     String sql = gen.generateSql("Top 5 employees by salary",
 *         "employees(id, name, dept, salary)");
 *     System.out.println(sql);
 * }
 * }</pre>
 *
 * @see TextGenerator
 * @see Summarizer
 * @see Translator
 * @see SqlGenerator
 * @see GrammarCorrector
 * @see GenerationResult
 */
public class FlanT5TextGenerator implements TextGenerator, Summarizer, Translator,
        SqlGenerator, GrammarCorrector {

    private final GenerationEngine engine;

    FlanT5TextGenerator(GenerationEngine engine) {
        this.engine = engine;
    }

    /**
     * Flan-T5 Small (77M parameters) preset.
     *
     * <p>Fast and lightweight. Downloads from {@code inference4j/flan-t5-small}
     * on first use.
     */
    public static Builder flanT5Small() {
        return builder().modelId("inference4j/flan-t5-small");
    }

    /**
     * Flan-T5 Base (250M parameters) preset.
     *
     * <p>Good balance of speed and quality. Downloads from
     * {@code inference4j/flan-t5-base} on first use.
     */
    public static Builder flanT5Base() {
        return builder().modelId("inference4j/flan-t5-base");
    }

    /**
     * Flan-T5 Large (780M parameters) preset.
     *
     * <p>Best quality. Downloads from {@code inference4j/flan-t5-large}
     * on first use. Requires external data files.
     */
    public static Builder flanT5Large() {
        return builder()
                .modelId("inference4j/flan-t5-large")
                .requiredFile("decoder_model.onnx_data")
                .requiredFile("encoder_model.onnx_data");
    }

    /**
     * Generic builder for custom Flan-T5 or compatible encoder-decoder models.
     *
     * <p>Requires at minimum a {@link Builder#modelId(String) modelId} (or
     * {@link Builder#modelSource(ModelSource) modelSource}) pointing to a directory
     * with {@code encoder_model.onnx}, {@code decoder_model.onnx},
     * {@code decoder_with_past_model.onnx}, and {@code config.json}.
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

    // --- Summarizer ---

    @Override
    public GenerationResult summarize(String text, Consumer<String> tokenListener) {
        return engine.generate("summarize: " + text, tokenListener);
    }

    // --- Translator ---

    @Override
    public GenerationResult translate(String text, Consumer<String> tokenListener) {
        throw new UnsupportedOperationException(
                "Flan-T5 requires source and target languages. "
                + "Use translate(text, source, target) instead.");
    }

    @Override
    public GenerationResult translate(String text, Language source, Language target,
                                       Consumer<String> tokenListener) {
        return engine.generate(
                "translate " + source.displayName() + " to " + target.displayName() + ": " + text,
                tokenListener);
    }

    // --- SqlGenerator ---

    @Override
    public GenerationResult generateSql(String query, String schema,
                                         Consumer<String> tokenListener) {
        return engine.generate(
                "generate SQL given the question and schema. question: " + query
                + " schema: " + schema,
                tokenListener);
    }

    // --- GrammarCorrector ---

    @Override
    public GenerationResult correct(String text, Consumer<String> tokenListener) {
        return engine.generate("correct grammar: " + text, tokenListener);
    }

    @Override
    public void close() throws Exception {
        engine.close();
    }

    public static class Builder
            extends AbstractEncoderDecoderBuilder<FlanT5TextGenerator, Builder> {

        @Override
        protected TokenizerProvider defaultTokenizerProvider() {
            return UnigramTokenizer.provider();
        }

        @Override
        protected FlanT5TextGenerator createWrapper(GenerationEngine engine) {
            return new FlanT5TextGenerator(engine);
        }
    }
}
