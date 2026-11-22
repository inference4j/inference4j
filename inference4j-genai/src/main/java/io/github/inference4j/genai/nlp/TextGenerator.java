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
import io.github.inference4j.generation.ChatMessage;
import io.github.inference4j.generation.ChatTemplate;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.generation.GenerativeModel;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.model.ResolvedModel;

import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

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
 * <p>A conversation, with a system message:
 * <pre>{@code
 * GenerationResult result = gen.chat(List.of(
 *         ChatMessage.system("Answer in one sentence."),
 *         ChatMessage.user("What is Java?")));
 * }</pre>
 *
 * <p>Limiting the answer length and threads, with greedy decoding:
 * <pre>{@code
 * try (var gen = TextGenerator.builder()
 *         .model(ModelSources.phi4Mini())
 *         .intraOpThreads(4)
 *         .maxNewTokens(200)
 *         .temperature(0)
 *         .stopSequence("\n\n")
 *         .build()) {
 *     System.out.println(gen.generate("Name three Java build tools.").text());
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
    private final int maxNewTokens;
    private final double repetitionPenalty;
    private final Set<String> stopSequences;

    TextGenerator(Model model, Tokenizer tokenizer, ChatTemplate chatTemplate,
                  int maxLength, double temperature, int topK, double topP) {
        this(model, tokenizer, chatTemplate, maxLength, temperature, topK, topP, 0, 0, Set.of());
    }

    TextGenerator(Model model, Tokenizer tokenizer, ChatTemplate chatTemplate,
                  int maxLength, double temperature, int topK, double topP,
                  int maxNewTokens, double repetitionPenalty, Set<String> stopSequences) {
        this(model, null, tokenizer, chatTemplate, maxLength, temperature, topK, topP, maxNewTokens,
                repetitionPenalty, stopSequences);
    }

    TextGenerator(Model model, ResolvedModel resolvedModel, Tokenizer tokenizer, ChatTemplate chatTemplate,
                  int maxLength, double temperature, int topK, double topP,
                  int maxNewTokens, double repetitionPenalty, Set<String> stopSequences) {
        super(model, resolvedModel);
        this.tokenizer = tokenizer;
        this.chatTemplate = chatTemplate;
        this.maxLength = maxLength;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.maxNewTokens = maxNewTokens;
        this.repetitionPenalty = repetitionPenalty;
        this.stopSequences = Set.copyOf(stopSequences);
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
        if (repetitionPenalty > 0) {
            params.setSearchOption("repetition_penalty", repetitionPenalty);
        }
        return params;
    }

    @Override
    protected Set<String> stopSequences() {
        return stopSequences;
    }


    /**
     * Generates the next assistant message of a conversation.
     *
     * @param messages the conversation, oldest message first
     * @return the generated text
     * @see #chat(List, Consumer)
     */
    public GenerationResult chat(List<ChatMessage> messages) {
        return chat(messages, null);
    }

    /**
     * Generates the next assistant message of a conversation, streaming each decoded fragment to
     * {@code tokenListener}. Thread-safe, with the same locking as {@link #generate(Object, Consumer)}.
     *
     * <p>The conversation is formatted with the chat template's {@link ChatTemplate#format(List)}, which the
     * templates of the models in {@code ModelSources} implement.
     *
     * @param messages      the conversation, oldest message first
     * @param tokenListener receives each decoded fragment; may be {@code null}
     * @return the generated text
     * @throws IllegalArgumentException if {@code messages} is empty
     * @throws UnsupportedOperationException if the chat template formats single user messages only
     */
    public GenerationResult chat(List<ChatMessage> messages, Consumer<String> tokenListener) {
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("messages must not be empty");
        }
        List<ChatMessage> conversation = List.copyOf(messages);
        return generateFrom(generator -> generator.appendTokenSequences(tokenizer.encode(format(conversation))),
                tokenListener);
    }

    /** The total number of tokens, prompt included, of one generation. */
    public int maxLength() {
        return maxLength;
    }

    /** The maximum number of tokens generated per call, or 0 when only {@link #maxLength()} applies. */
    public int maxNewTokens() {
        return maxNewTokens;
    }

    private String format(List<ChatMessage> messages) {
        try {
            return chatTemplate.format(messages);
        } catch (UnsupportedOperationException e) {
            throw new UnsupportedOperationException("The chat template formats single user messages only. Use a "
                    + "model from ModelSources, or a ChatTemplate that implements format(List<ChatMessage>)", e);
        }
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
    protected GenerationResult parseOutput(String generatedText, String input, int promptTokens,
                                           int tokenCount, long durationMillis) {
        return new GenerationResult(generatedText, promptTokens, tokenCount,
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
        private int intraOpThreads;
        private int maxNewTokens;
        private double repetitionPenalty;
        private final Set<String> stopSequences = new LinkedHashSet<>();

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

        /**
         * The total number of tokens, prompt included, the model may hold in one generation. Defaults to 1024.
         * The KV cache is sized for it. To limit only the answer, use {@link #maxNewTokens(int)}.
         */
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

        /**
         * Limits each ONNX Runtime session of the model to {@code intraOpThreads} threads. Defaults to ONNX
         * Runtime's choice, which uses every core.
         */
        public Builder intraOpThreads(int intraOpThreads) {
            if (intraOpThreads < 1) {
                throw new IllegalArgumentException("intraOpThreads must be at least 1");
            }
            this.intraOpThreads = intraOpThreads;
            return this;
        }

        /**
         * The maximum number of tokens to generate per call, not counting the prompt. Generation also stops at
         * {@link #maxLength(int)}, whichever comes first. Defaults to no limit of its own.
         */
        public Builder maxNewTokens(int maxNewTokens) {
            if (maxNewTokens < 1) {
                throw new IllegalArgumentException("maxNewTokens must be at least 1");
            }
            this.maxNewTokens = maxNewTokens;
            return this;
        }

        /**
         * Penalises tokens that were already generated; values above 1 make repetition less likely. Defaults to
         * the model's own setting.
         */
        public Builder repetitionPenalty(double repetitionPenalty) {
            if (repetitionPenalty <= 0) {
                throw new IllegalArgumentException("repetitionPenalty must be positive");
            }
            this.repetitionPenalty = repetitionPenalty;
            return this;
        }

        /**
         * Adds a string that ends generation when it appears in the output. The stop sequence and anything after
         * it are left out of the result and are not streamed. Can be called several times.
         */
        public Builder stopSequence(String stopSequence) {
            if (stopSequence == null || stopSequence.isEmpty()) {
                throw new IllegalArgumentException("stopSequence must not be empty");
            }
            this.stopSequences.add(stopSequence);
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
                ResolvedModel resolved = modelSource.open("model");
                Model m = null;
                try {
                    m = loadModel(resolved.directory(), intraOpThreads);
                    Tokenizer t = new Tokenizer(m);
                    return new TextGenerator(m, resolved, t, chatTemplate,
                            maxLength, temperature, topK, topP, maxNewTokens, repetitionPenalty, stopSequences);
                } catch (GenAIException | IOException e) {
                    if (m != null) {
                        m.close();
                    }
                    resolved.close();
                    throw new ModelSourceException(
                            "Failed to load genai model: " + e.getMessage(), e);
                }
            }
            return new TextGenerator(model, tokenizer, chatTemplate,
                    maxLength, temperature, topK, topP, maxNewTokens, repetitionPenalty, stopSequences);
        }
    }
}
