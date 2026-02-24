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
import io.github.inference4j.tokenizer.DecodingBpeTokenizer;
import io.github.inference4j.tokenizer.SentencePieceBpeTokenizer;
import io.github.inference4j.tokenizer.TokenDecoder;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.TokenizerProvider;

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
 * // TinyLlama-1.1B-Chat — Zephyr-style instruct model
 * try (var gen = OnnxTextGenerator.tinyLlama().maxNewTokens(100).build()) {
 *     System.out.println(gen.generate("Explain gravity").text());
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
                .stopSequence("<|im_end|>")
                .chatTemplate(msg ->
                        "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n");
    }

    /**
     * SmolLM2-1.7B-Instruct preset.
     *
     * <p>ChatML instruct model (FP16). Downloads from
     * {@code inference4j/smollm2-1.7b-instruct} (~3.4 GB) on first use.
     */
    public static Builder smolLM2_1_7B() {
        return builder()
                .modelId("inference4j/smollm2-1.7b-instruct")
                .requiredFile("model.onnx_data")
                .addedToken("<|im_start|>")
                .addedToken("<|im_end|>")
                .addedToken("<|endoftext|>")
                .stopSequence("<|im_end|>")
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
                .requiredFile("model.onnx_data")
                .addedToken("<|im_start|>")
                .addedToken("<|im_end|>")
                .addedToken("<|endoftext|>")
                .stopSequence("<|im_end|>")
                .tokenizerProvider(DecodingBpeTokenizer.provider(QWEN2_PATTERN))
                .chatTemplate(msg ->
                        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                                + "<|im_start|>user\n" + msg + "<|im_end|>\n"
                                + "<|im_start|>assistant\n");
    }

    /**
     * TinyLlama-1.1B-Chat preset.
     *
     * <p>Zephyr-style instruct model using SentencePiece tokenization. Downloads from
     * {@code inference4j/tinyllama-1.1b-chat} (~2.2 GB) on first use.
     */
    public static Builder tinyLlama() {
        return builder()
                .modelId("inference4j/tinyllama-1.1b-chat")
                .requiredFile("model.onnx_data")
                .tokenizerProvider(SentencePieceBpeTokenizer.provider())
                .addedToken("<|user|>")
                .addedToken("<|assistant|>")
                .addedToken("<|system|>")
                .eosTokenId(2)          // </s>
                .stopSequence("</s>")
                .chatTemplate(msg ->
                        "<|user|>\n" + msg + "</s>\n<|assistant|>\n");
    }

    /**
     * Gemma 2-2B-IT preset.
     *
     * <p>Instruction-tuned model using SentencePiece tokenization.
     *
     * <p><strong>Note:</strong> Gemma is a gated model. You must accept Google's
     * license terms on HuggingFace, download the ONNX model yourself, and provide
     * it via {@link Builder#modelSource(ModelSource) modelSource}:
     *
     * <pre>{@code
     * try (var gen = OnnxTextGenerator.gemma2()
     *         .modelSource(id -> Path.of("/path/to/gemma-2-2b-it"))
     *         .maxNewTokens(100)
     *         .build()) {
     *     gen.generate("What is Java?", token -> System.out.print(token));
     * }
     * }</pre>
     *
     * @see <a href="https://huggingface.co/google/gemma-2-2b-it">Gemma 2-2B-IT on HuggingFace</a>
     */
    public static Builder gemma2() {
        return builder()
                .requiredFile("model.onnx_data")
                .tokenizerProvider(SentencePieceBpeTokenizer.provider())
                .addedToken("<start_of_turn>")
                .addedToken("<end_of_turn>")
                .addedToken("<bos>")
                .addedToken("<eos>")
                .eosTokenId(1)
                .eosTokenId(107)
                .stopSequence("<end_of_turn>")
                .chatTemplate(msg ->
                        "<bos><start_of_turn>user\n" + msg
                                + "<end_of_turn>\n<start_of_turn>model\n");
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
        private TokenizerProvider tokenizerProvider;
        private int maxNewTokens = 256;
        private float temperature = 0f;
        private int topK = 0;
        private float topP = 0f;
        private final Set<Integer> eosTokenIds = new LinkedHashSet<>();
        private final Set<String> stopSequences = new LinkedHashSet<>();
        private final List<String> addedTokens = new ArrayList<>();
        private final List<String> extraFiles = new ArrayList<>();

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

        public Builder tokenizerProvider(TokenizerProvider tokenizerProvider) {
            this.tokenizerProvider = tokenizerProvider;
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

        Builder requiredFile(String filename) {
            this.extraFiles.add(filename);
            return this;
        }

        public OnnxTextGenerator build() {
            if (modelId == null && modelSource == null) {
                throw new ModelLoadException(
                        "Model source not configured. Set modelId() for HuggingFace models "
                        + "or modelSource() for local models.");
            }
            ModelSource source = modelSource != null
                    ? modelSource : HuggingFaceModelSource.defaultInstance();
            String id = modelId != null ? modelId : "";
            TokenizerProvider provider = tokenizerProvider != null
                    ? tokenizerProvider : DecodingBpeTokenizer.provider();
            List<String> files = new ArrayList<>(List.of("model.onnx", "config.json"));
            files.addAll(provider.requiredFiles());
            files.addAll(extraFiles);
            Path dir = source.resolve(id, files);
            return loadFromDirectory(dir, provider);
        }

        private OnnxTextGenerator loadFromDirectory(Path dir, TokenizerProvider provider) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            Path configPath = dir.resolve("config.json");

            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
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
                    TokenizerProvider.TokenizerAndDecoder td =
                            provider.create(dir, addedTokens);
                    if (this.tokenizer == null) {
                        this.tokenizer = td.tokenizer();
                    }
                    if (this.decoder == null) {
                        this.decoder = td.decoder();
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
