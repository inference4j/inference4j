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
package io.github.inference4j.genai.nlp;

import ai.onnxruntime.genai.GenAIException;
import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.Sequences;
import ai.onnxruntime.genai.Tokenizer;
import ai.onnxruntime.genai.TokenizerStream;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.genai.AbstractGenerativeTask;
import io.github.inference4j.generation.ChatTemplate;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.generation.GenerativeModel;
import io.github.inference4j.model.ModelSource;

import java.nio.file.Path;

/**
 * Autoregressive text generator backed by onnxruntime-genai.
 *
 * <p>Wraps decoder-only language models (Phi-3, GPT-2, Llama, etc.) with an
 * ergonomic builder API. The model's chat template is applied to format
 * prompts before tokenization.
 *
 * <p>Usage:
 * <pre>{@code
 * try (var gen = TextGenerator.builder()
 *         .model(ModelSources.phi3Mini())
 *         .build()) {
 *     GenerationResult result = gen.generate("What is Java in one sentence?");
 *     System.out.println(result.text());
 * }
 * }</pre>
 *
 * <p>With streaming:
 * <pre>{@code
 * try (var gen = TextGenerator.builder()
 *         .model(ModelSources.phi3Mini())
 *         .maxLength(200)
 *         .temperature(0.7)
 *         .build()) {
 *     gen.generate("Explain recursion.", token -> System.out.print(token));
 * }
 * }</pre>
 *
 * @see io.github.inference4j.genai.ModelSources
 */
public class TextGenerator extends AbstractGenerativeTask<String, GenerationResult> {

    private final Tokenizer tokenizer;
    private final ChatTemplate chatTemplate;
    private final int maxLength;
    private final double temperature;
    private final int topK;
    private final double topP;

    TextGenerator(Model model, Tokenizer tokenizer, ChatTemplate chatTemplate,
                  int maxLength, double temperature, int topK, double topP) {
        super(model);
        this.tokenizer = tokenizer;
        this.chatTemplate = chatTemplate;
        this.maxLength = maxLength;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
    }

    @Override
    protected GeneratorParams createParams() throws GenAIException {
        GeneratorParams params = super.createParams();
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
    protected TokenizerStream createStream() throws GenAIException {
        return tokenizer.createStream();
    }

    @Override
    protected void prepareGenerator(String input, Generator generator) {
        try {
            String formatted = chatTemplate.format(input);
            Sequences sequences = tokenizer.encode(formatted);
            generator.appendTokenSequences(sequences);
        } catch (GenAIException e) {
            throw new InferenceException(
                    "Failed to prepare generator input: " + e.getMessage(), e);
        }
    }

    @Override
    protected GenerationResult parseOutput(String generatedText, String input,
                                           int tokenCount, long durationMillis) {
        return new GenerationResult(generatedText, 0, tokenCount,
                java.time.Duration.ofMillis(durationMillis));
    }

    @Override
    protected void closeResources() {
        tokenizer.close();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private ModelSource modelSource;
        private ChatTemplate chatTemplate;
        private int maxLength = 1024;
        private double temperature = 1.0;
        private int topK = 0;
        private double topP = 0.0;

        // Package-private for testing
        Model model;
        Tokenizer tokenizer;

        public Builder model(GenerativeModel generativeModel) {
            this.modelSource = generativeModel.modelSource();
            this.chatTemplate = generativeModel.chatTemplate();
            return this;
        }

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder chatTemplate(ChatTemplate chatTemplate) {
            this.chatTemplate = chatTemplate;
            return this;
        }

        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        public Builder temperature(double temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }

        public Builder topP(double topP) {
            this.topP = topP;
            return this;
        }

        public TextGenerator build() {
            if (model == null) {
                if (modelSource == null) {
                    throw new IllegalStateException(
                            "model is required — use model(ModelSources.phi3Mini()) "
                                    + "or provide modelSource + chatTemplate");
                }
                if (chatTemplate == null) {
                    throw new IllegalStateException(
                            "chatTemplate is required — use model(ModelSources.phi3Mini()) "
                                    + "or provide a chatTemplate alongside modelSource");
                }
                try {
                    Path modelDir = modelSource.resolve("model");
                    model = new Model(modelDir.toString());
                    tokenizer = new Tokenizer(model);
                } catch (GenAIException e) {
                    throw new ModelSourceException(
                            "Failed to load genai model: " + e.getMessage(), e);
                }
            }
            return new TextGenerator(model, tokenizer, chatTemplate,
                    maxLength, temperature, topK, topP);
        }
    }
}
