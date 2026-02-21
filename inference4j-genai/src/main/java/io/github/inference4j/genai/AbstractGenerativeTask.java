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
package io.github.inference4j.genai;

import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.TokenizerStream;
import ai.onnxruntime.genai.GenAIException;

import io.github.inference4j.generation.GenerativeTask;

import java.util.function.Consumer;

/**
 * Base class for autoregressive generation tasks backed by onnxruntime-genai.
 *
 * <p>This is the generative counterpart to
 * {@link io.github.inference4j.AbstractInferenceTask AbstractInferenceTask}.
 * While AbstractInferenceTask enforces a preprocess &rarr; infer &rarr; postprocess pipeline
 * for single-pass models, this class owns the autoregressive generate loop:
 * prepare generator &rarr; generate tokens one by one &rarr; parse output.
 *
 * <p>Subclasses provide two hooks:
 * <ul>
 *   <li>{@link #prepareGenerator(Object, Generator)} &mdash; feed the generator with
 *       encoded input (text tokens, audio tensors, image tensors)</li>
 *   <li>{@link #parseOutput(String, Object, int, long)} &mdash; convert the generated
 *       text into the domain output type</li>
 * </ul>
 *
 * <p>The generate loop itself is {@code final} and cannot be overridden.
 *
 * @param <I> the domain input type
 * @param <O> the domain output type
 */
public abstract class AbstractGenerativeTask<I, O> implements GenerativeTask<I, O> {

    protected final Model model;

    protected AbstractGenerativeTask(Model model) {
        this.model = model;
    }

    @Override
    public final O generate(I input) {
        return generate(input, null);
    }

    @Override
    public final O generate(I input, Consumer<String> tokenListener) {
        try {
            try (GeneratorParams params = createParams();
                 Generator generator = new Generator(model, params)) {
                prepareGenerator(input, generator);

                long start = System.currentTimeMillis();
                int tokenCount = 0;
                StringBuilder sb = new StringBuilder();

                try (TokenizerStream stream = createStream()) {
                    while (!generator.isDone()) {
                        generator.generateNextToken();
                        int token = generator.getLastTokenInSequence(0);
                        String text = stream.decode(token);
                        sb.append(text);
                        tokenCount++;
                        if (tokenListener != null && !text.isEmpty()) {
                            tokenListener.accept(text);
                        }
                    }
                }

                long duration = System.currentTimeMillis() - start;
                return parseOutput(sb.toString(), input, tokenCount, duration);
            }
        } catch (GenAIException e) {
            throw new io.github.inference4j.exception.InferenceException(
                    "Generation failed: " + e.getMessage(), e);
        }
    }

    /**
     * Create a {@link TokenizerStream} for decoding generated tokens into text.
     *
     * <p>Subclasses backed by a {@code Tokenizer} (e.g. text generators) create the
     * stream from the tokenizer. Subclasses backed by a {@code MultiModalProcessor}
     * (e.g. Whisper audio transcription) create the stream from the processor.
     *
     * @return a new TokenizerStream for the current generation
     * @throws GenAIException if stream creation fails
     */
    protected abstract TokenizerStream createStream() throws GenAIException;

    /**
     * Feed the generator with encoded input before the generate loop starts.
     *
     * <p>Subclasses typically encode the input into token IDs and append them
     * to the generator's input sequence.
     *
     * @param input     the domain input
     * @param generator the generator to prepare
     */
    protected abstract void prepareGenerator(I input, Generator generator);

    /**
     * Convert the generated text and metadata into the domain output type.
     *
     * @param generatedText  the full generated text
     * @param input          the original domain input
     * @param tokenCount     number of tokens generated
     * @param durationMillis wall-clock generation time in milliseconds
     * @return the domain output
     */
    protected abstract O parseOutput(String generatedText, I input,
                                     int tokenCount, long durationMillis);

    /**
     * Create {@link GeneratorParams} for this generation run.
     *
     * <p>The default implementation returns bare params with no search options.
     * Subclasses should override this to configure model-specific options
     * (max_length, temperature, top_k, top_p, etc.).
     *
     * @return configured generator params
     * @throws GenAIException if param creation fails
     */
    protected GeneratorParams createParams() throws GenAIException {
        return new GeneratorParams(model);
    }

    /**
     * Release subclass-owned resources (e.g. Tokenizer, MultiModalProcessor).
     *
     * <p>Called by {@link #close()} before closing the model. Subclasses should
     * override this to close any resources they own. The default implementation
     * does nothing.
     */
    protected void closeResources() {
    }

    @Override
    public final void close() {
        closeResources();
        model.close();
    }
}
