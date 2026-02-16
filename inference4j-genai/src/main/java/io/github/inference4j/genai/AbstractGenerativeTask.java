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
import ai.onnxruntime.genai.Tokenizer;
import ai.onnxruntime.genai.TokenizerStream;
import ai.onnxruntime.genai.GenAIException;

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
    protected final Tokenizer tokenizer;
    private final int maxLength;
    private final double temperature;
    private final int topK;
    private final double topP;

    protected AbstractGenerativeTask(Model model, Tokenizer tokenizer,
                                     int maxLength, double temperature,
                                     int topK, double topP) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.maxLength = maxLength;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
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

                try (TokenizerStream stream = tokenizer.createStream()) {
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

    private GeneratorParams createParams() throws GenAIException {
        GeneratorParams params = new GeneratorParams(model);
        params.setSearchOption("max_length", maxLength);
        if (temperature > 0) {
            params.setSearchOption("temperature", temperature);
        }
        if (topK > 0) {
            params.setSearchOption("top_k", topK);
        }
        if (topP > 0) {
            params.setSearchOption("top_p", topP);
        }
        return params;
    }

    @Override
    public void close() {
        tokenizer.close();
        model.close();
    }
}
