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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.generation.ChatTemplate;
import io.github.inference4j.generation.GenerationEngine;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.generation.OnnxGenerativeSession;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.tokenizer.BpeTokenizer;
import io.github.inference4j.tokenizer.DecodingBpeTokenizer;
import io.github.inference4j.tokenizer.TokenDecoder;
import io.github.inference4j.tokenizer.Tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Pattern;

/**
 * General-purpose ONNX text generator for autoregressive language models.
 *
 * <p>Works with any causal language model exported to ONNX with KV cache
 * support and BPE tokenization ({@code vocab.json} + {@code merges.txt}).
 * Named presets provide one-liner access to popular models; the generic
 * {@link #builder()} supports custom models.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * // GPT-2 (124M) — completion model
 * try (var gen = OnnxTextGenerator.gpt2().build()) {
 *     System.out.println(gen.generate("Once upon a time").text());
 * }
 *
 * // SmolLM2-360M-Instruct — ChatML instruct model
 * try (var gen = OnnxTextGenerator.smolLM2().build()) {
 *     System.out.println(gen.generate("What is the capital of France?").text());
 * }
 *
 * // Qwen2.5-1.5B-Instruct — ChatML instruct model
 * try (var gen = OnnxTextGenerator.qwen2().maxNewTokens(100).build()) {
 *     System.out.println(gen.generate("Explain gravity").text());
 * }
 * }</pre>
 *
 * <h2>Custom model</h2>
 * <pre>{@code
 * try (var gen = OnnxTextGenerator.builder()
 *         .modelId("my-org/my-model")
 *         .addedToken("<|special|>")
 *         .chatTemplate(msg -> "<|user|>" + msg + "<|assistant|>")
 *         .temperature(0.7f)
 *         .build()) {
 *     gen.generate("Hello", token -> System.out.print(token));
 * }
 * }</pre>
 *
 * @see TextGenerator
 * @see GenerationResult
 */
public class OnnxTextGenerator implements TextGenerator {

    /**
     * Qwen2 / Qwen2.5 pre-tokenization pattern. Differs from GPT-2 in
     * case-insensitive contractions, single-digit matching, and newline handling.
     */
    static final Pattern QWEN2_PATTERN = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    );

    private final GenerationEngine engine;

    private OnnxTextGenerator(GenerationEngine engine) {
        this.engine = engine;
    }

    /**
     * GPT-2 (124M parameters) preset.
     *
     * <p>Completion model with no chat template. Downloads from
     * {@code inference4j/gpt2} (~500 MB) on first use.
     */
    public static Builder gpt2() {
        return builder().modelId("inference4j/gpt2");
    }

    /**
     * SmolLM2-360M-Instruct preset.
     *
     * <p>ChatML instruct model. Downloads from
     * {@code inference4j/smollm2-360m-instruct} (~700 MB) on first use.
     */
    public static Builder smolLM2() {
        return builder()
                .modelId("inference4j/smollm2-360m-instruct")
                .addedToken("<|im_start|>")
                .addedToken("<|im_end|>")
                .addedToken("<|endoftext|>")
                .chatTemplate(msg ->
                        "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n");
    }

    /**
     * Qwen2.5-1.5B-Instruct preset.
     *
     * <p>ChatML instruct model with system prompt. Downloads from
     * {@code inference4j/qwen2.5-1.5b-instruct} (~3 GB) on first use.
     */
    public static Builder qwen2() {
        return builder()
                .modelId("inference4j/qwen2.5-1.5b-instruct")
                .addedToken("<|im_start|>")
                .addedToken("<|im_end|>")
                .addedToken("<|endoftext|>")
                .tokenizerPattern(QWEN2_PATTERN)
                .chatTemplate(msg ->
                        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                                + "<|im_start|>user\n" + msg + "<|im_end|>\n"
                                + "<|im_start|>assistant\n");
    }

    /**
     * Generic builder for custom models.
     *
     * <p>Requires at minimum a {@link #modelId(String) modelId} (or
     * {@link #modelSource(ModelSource) modelSource}) pointing to a directory
     * with {@code model.onnx}, {@code vocab.json}, {@code merges.txt}, and
     * {@code config.json}.
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

    @Override
    public void close() throws Exception {
        engine.close();
    }

    public static class Builder {

        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private Tokenizer tokenizer;
        private TokenDecoder decoder;
        private ChatTemplate chatTemplate;
        private Pattern tokenizerPattern;
        private int maxNewTokens = 256;
        private float temperature = 0f;
        private int topK = 0;
        private float topP = 0f;
        private final Set<Integer> eosTokenIds = new LinkedHashSet<>();
        private final Set<String> stopSequences = new LinkedHashSet<>();
        private final List<String> addedTokens = new ArrayList<>();

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public Builder tokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder decoder(TokenDecoder decoder) {
            this.decoder = decoder;
            return this;
        }

        public Builder chatTemplate(ChatTemplate chatTemplate) {
            this.chatTemplate = chatTemplate;
            return this;
        }

        public Builder tokenizerPattern(Pattern pattern) {
            this.tokenizerPattern = pattern;
            return this;
        }

        public Builder maxNewTokens(int maxNewTokens) {
            this.maxNewTokens = maxNewTokens;
            return this;
        }

        public Builder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }

        public Builder topP(float topP) {
            this.topP = topP;
            return this;
        }

        public Builder eosTokenId(int eosTokenId) {
            this.eosTokenIds.add(eosTokenId);
            return this;
        }

        public Builder stopSequence(String stopSequence) {
            this.stopSequences.add(stopSequence);
            return this;
        }

        public Builder addedToken(String token) {
            this.addedTokens.add(token);
            return this;
        }

        public OnnxTextGenerator build() {
            ModelSource source = modelSource != null
                    ? modelSource : HuggingFaceModelSource.defaultInstance();
            String id = modelId != null ? modelId : "inference4j/gpt2";
            Path dir = source.resolve(id,
                    List.of("model.onnx", "vocab.json", "merges.txt", "config.json"));
            return loadFromDirectory(dir);
        }

        private OnnxTextGenerator loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            Path vocabPath = dir.resolve("vocab.json");
            Path mergesPath = dir.resolve("merges.txt");
            Path configPath = dir.resolve("config.json");

            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }
            if (!Files.exists(vocabPath)) {
                throw new ModelSourceException("Vocabulary file not found: " + vocabPath);
            }
            if (!Files.exists(mergesPath)) {
                throw new ModelSourceException("Merges file not found: " + mergesPath);
            }
            if (!Files.exists(configPath)) {
                throw new ModelSourceException("Config file not found: " + configPath);
            }

            InferenceSession session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);

            try {
                OnnxGenerativeSession generativeSession = new OnnxGenerativeSession(session);

                if (this.tokenizer == null || this.decoder == null) {
                    BpeTokenizer.Builder bpeBuilder =
                            BpeTokenizer.builder(vocabPath, mergesPath);
                    for (String token : addedTokens) {
                        bpeBuilder.addedToken(token);
                    }
                    if (tokenizerPattern != null) {
                        bpeBuilder.pattern(tokenizerPattern);
                    }
                    DecodingBpeTokenizer bpe = DecodingBpeTokenizer.from(bpeBuilder);
                    if (this.tokenizer == null) {
                        this.tokenizer = bpe;
                    }
                    if (this.decoder == null) {
                        this.decoder = bpe;
                    }
                }

                Set<Integer> eos = this.eosTokenIds.isEmpty()
                        ? readEosTokenIds(configPath)
                        : this.eosTokenIds;

                GenerationEngine.Builder engineBuilder = GenerationEngine.builder()
                        .session(generativeSession)
                        .tokenizer(this.tokenizer)
                        .decoder(this.decoder)
                        .maxNewTokens(this.maxNewTokens)
                        .temperature(this.temperature)
                        .topK(this.topK)
                        .topP(this.topP);

                for (int eosId : eos) {
                    engineBuilder.eosTokenId(eosId);
                }
                if (this.chatTemplate != null) {
                    engineBuilder.chatTemplate(this.chatTemplate);
                }
                for (String seq : this.stopSequences) {
                    engineBuilder.stopSequence(seq);
                }

                return new OnnxTextGenerator(engineBuilder.build());
            } catch (Exception e) {
                session.close();
                if (e instanceof RuntimeException re) {
                    throw re;
                }
                throw new ModelLoadException(
                        "Failed to initialize model: " + e.getMessage(), e);
            }
        }

        private static Set<Integer> readEosTokenIds(Path configPath) {
            try {
                ObjectMapper mapper = new ObjectMapper();
                JsonNode root = mapper.readTree(Files.newInputStream(configPath));
                JsonNode eosNode = root.get("eos_token_id");
                if (eosNode != null) {
                    if (eosNode.isInt()) {
                        return Set.of(eosNode.intValue());
                    }
                    if (eosNode.isArray()) {
                        Set<Integer> ids = new LinkedHashSet<>();
                        for (JsonNode element : eosNode) {
                            if (element.isInt()) {
                                ids.add(element.intValue());
                            }
                        }
                        if (!ids.isEmpty()) {
                            return ids;
                        }
                    }
                }
                throw new ModelLoadException(
                        "config.json missing eos_token_id — set it explicitly via eosTokenId()");
            } catch (IOException e) {
                throw new ModelLoadException(
                        "Failed to read config.json: " + e.getMessage(), e);
            }
        }
    }
}
