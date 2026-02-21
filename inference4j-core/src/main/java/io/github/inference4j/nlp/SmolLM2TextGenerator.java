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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

/**
 * SmolLM2-360M-Instruct text generator.
 *
 * <h2>Target model</h2>
 * <p>Designed for <a href="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct">SmolLM2-360M-Instruct</a>
 * (360M parameters, Apache 2.0 license). Works with any SmolLM2 variant exported to ONNX
 * with KV cache support using BPE tokenization and ChatML format.
 *
 * <p>The model directory should contain:
 * <ul>
 *   <li>{@code model.onnx} — the ONNX model with KV cache inputs/outputs</li>
 *   <li>{@code vocab.json} — BPE vocabulary</li>
 *   <li>{@code merges.txt} — BPE merge rules</li>
 *   <li>{@code config.json} — model config with {@code eos_token_id}</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (SmolLM2TextGenerator gen = SmolLM2TextGenerator.builder().build()) {
 *     GenerationResult result = gen.generate("What is the capital of France?");
 *     System.out.println(result.text());
 * }
 * }</pre>
 *
 * <h2>Streaming</h2>
 * <pre>{@code
 * try (SmolLM2TextGenerator gen = SmolLM2TextGenerator.builder()
 *         .temperature(0.8f)
 *         .topP(0.9f)
 *         .maxNewTokens(100)
 *         .build()) {
 *     gen.generate("Explain quantum computing", token -> System.out.print(token));
 * }
 * }</pre>
 *
 * @see TextGenerator
 * @see GenerationResult
 */
public class SmolLM2TextGenerator implements TextGenerator {

    private static final String DEFAULT_MODEL_ID = "inference4j/smollm2-360m-instruct";
    private static final int DEFAULT_EOS_TOKEN_ID = 2;
    private static final ChatTemplate DEFAULT_CHAT_TEMPLATE =
            msg -> "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";

    private final GenerationEngine engine;

    private SmolLM2TextGenerator(GenerationEngine engine) {
        this.engine = engine;
    }

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
        private int maxNewTokens = 256;
        private float temperature = 0f;
        private int topK = 0;
        private float topP = 0f;
        private int eosTokenId = -1;
        private final Set<String> stopSequences = new LinkedHashSet<>();

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
            this.eosTokenId = eosTokenId;
            return this;
        }

        public Builder stopSequence(String stopSequence) {
            this.stopSequences.add(stopSequence);
            return this;
        }

        public SmolLM2TextGenerator build() {
            ModelSource source = modelSource != null
                    ? modelSource : HuggingFaceModelSource.defaultInstance();
            String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
            Path dir = source.resolve(id,
                    List.of("model.onnx", "vocab.json", "merges.txt", "config.json"));
            return loadFromDirectory(dir);
        }

        private SmolLM2TextGenerator loadFromDirectory(Path dir) {
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
                    DecodingBpeTokenizer bpe = DecodingBpeTokenizer.from(
                            BpeTokenizer.builder(vocabPath, mergesPath)
                                    .addedToken("<|im_start|>")
                                    .addedToken("<|im_end|>")
                                    .addedToken("<|endoftext|>"));
                    if (this.tokenizer == null) {
                        this.tokenizer = bpe;
                    }
                    if (this.decoder == null) {
                        this.decoder = bpe;
                    }
                }

                int eos = this.eosTokenId >= 0
                        ? this.eosTokenId
                        : readEosTokenId(configPath);

                GenerationEngine.Builder engineBuilder = GenerationEngine.builder()
                        .session(generativeSession)
                        .tokenizer(this.tokenizer)
                        .decoder(this.decoder)
                        .eosTokenId(eos)
                        .maxNewTokens(this.maxNewTokens)
                        .temperature(this.temperature)
                        .topK(this.topK)
                        .topP(this.topP);

                ChatTemplate template = this.chatTemplate != null
                        ? this.chatTemplate : DEFAULT_CHAT_TEMPLATE;
                engineBuilder.chatTemplate(template);

                for (String seq : this.stopSequences) {
                    engineBuilder.stopSequence(seq);
                }

                return new SmolLM2TextGenerator(engineBuilder.build());
            } catch (Exception e) {
                session.close();
                if (e instanceof RuntimeException re) {
                    throw re;
                }
                throw new ModelLoadException(
                        "Failed to initialize SmolLM2 model: " + e.getMessage(), e);
            }
        }

        private static int readEosTokenId(Path configPath) {
            try {
                ObjectMapper mapper = new ObjectMapper();
                JsonNode root = mapper.readTree(Files.newInputStream(configPath));
                JsonNode eosNode = root.get("eos_token_id");
                if (eosNode != null && eosNode.isInt()) {
                    return eosNode.intValue();
                }
                return DEFAULT_EOS_TOKEN_ID;
            } catch (IOException e) {
                return DEFAULT_EOS_TOKEN_ID;
            }
        }
    }
}
