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

import ai.onnxruntime.genai.GenAIException;
import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.Sequences;
import ai.onnxruntime.genai.Tokenizer;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.genai.AbstractGenerativeTask;
import io.github.inference4j.genai.GenerationResult;
import io.github.inference4j.model.ModelSource;

import java.nio.file.Path;

/**
 * Autoregressive text generator backed by onnxruntime-genai.
 *
 * <p>Wraps decoder-only language models (Phi-3, GPT-2, Llama, etc.) with an
 * ergonomic builder API. The model's built-in chat template is used to format
 * prompts automatically.
 *
 * <p>Usage:
 * <pre>{@code
 * try (var gen = TextGenerator.builder()
 *         .modelSource(ModelSources.phi3Mini())
 *         .build()) {
 *     GenerationResult result = gen.generate("What is Java in one sentence?");
 *     System.out.println(result.text());
 * }
 * }</pre>
 *
 * <p>With streaming:
 * <pre>{@code
 * try (var gen = TextGenerator.builder()
 *         .modelSource(ModelSources.phi3Mini())
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

    TextGenerator(Model model, Tokenizer tokenizer,
                  int maxLength, double temperature, int topK, double topP) {
        super(model, tokenizer, maxLength, temperature, topK, topP);
    }

    @Override
    protected void prepareGenerator(String input, Generator generator) {
        try {
            String formatted = tokenizer.applyChatTemplate(
                    null,
                    buildMessagesJson(input),
                    null,
                    true);
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
        return new GenerationResult(generatedText, tokenCount, durationMillis);
    }

    String buildMessagesJson(String userMessage) {
        return "[{\"role\": \"user\", \"content\": \"%s\"}]"
                .formatted(userMessage.replace("\\", "\\\\").replace("\"", "\\\""));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private ModelSource modelSource;
        private int maxLength = 1024;
        private double temperature = 1.0;
        private int topK = 0;
        private double topP = 0.0;

        // Package-private for testing
        Model model;
        Tokenizer tokenizer;

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
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
                            "modelSource is required — use ModelSources.phi3Mini() "
                                    + "or provide a custom ModelSource");
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
            return new TextGenerator(model, tokenizer,
                    maxLength, temperature, topK, topP);
        }
    }
}
