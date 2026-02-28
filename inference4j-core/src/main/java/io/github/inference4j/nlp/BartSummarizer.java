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
import io.github.inference4j.tokenizer.DecodingBpeTokenizer;
import io.github.inference4j.tokenizer.TokenizerProvider;

import java.util.function.Consumer;

/**
 * Text summarizer backed by BART encoder-decoder models.
 *
 * <p>BART is a denoising autoencoder for pretraining sequence-to-sequence models.
 * Unlike Flan-T5, BART does not use task prefixes — input text is passed directly
 * to the encoder. This class implements {@link TextGenerator} for raw generation
 * and {@link Summarizer} for the summarization task.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * // DistilBART CNN (306M) — distilled, fast summarization
 * try (var summarizer = BartSummarizer.distilBartCnn().build()) {
 *     String summary = summarizer.summarize("Long article text...");
 *     System.out.println(summary);
 * }
 *
 * // BART Large CNN (406M) — best quality summarization
 * try (var summarizer = BartSummarizer.bartLargeCnn().build()) {
 *     String summary = summarizer.summarize("Long article text...");
 *     System.out.println(summary);
 * }
 * }</pre>
 *
 * @see TextGenerator
 * @see Summarizer
 * @see GenerationResult
 */
public class BartSummarizer implements TextGenerator, Summarizer {

    private final GenerationEngine engine;

    BartSummarizer(GenerationEngine engine) {
        this.engine = engine;
    }

    /**
     * DistilBART CNN (306M parameters) preset.
     *
     * <p>Distilled and fast. Downloads from {@code inference4j/distilbart-cnn-12-6}
     * on first use.
     */
    public static Builder distilBartCnn() {
        return builder().modelId("inference4j/distilbart-cnn-12-6");
    }

    /**
     * BART Large CNN (406M parameters) preset.
     *
     * <p>Best quality summarization. Downloads from {@code inference4j/bart-large-cnn}
     * on first use. Requires external data files.
     */
    public static Builder bartLargeCnn() {
        return builder()
                .modelId("inference4j/bart-large-cnn")
                .requiredFile("decoder_model.onnx_data")
                .requiredFile("encoder_model.onnx_data");
    }

    /**
     * Generic builder for custom BART or compatible encoder-decoder models.
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

    // --- Summarizer (no prefix — BART takes raw text) ---

    @Override
    public GenerationResult summarize(String text, Consumer<String> tokenListener) {
        return engine.generate(text, tokenListener);
    }

    @Override
    public void close() throws Exception {
        engine.close();
    }

    public static class Builder
            extends AbstractEncoderDecoderBuilder<BartSummarizer, Builder> {

        @Override
        protected TokenizerProvider defaultTokenizerProvider() {
            return DecodingBpeTokenizer.provider();
        }

        @Override
        protected BartSummarizer createWrapper(GenerationEngine engine) {
            return new BartSummarizer(engine);
        }
    }
}
