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
import io.github.inference4j.generation.EncoderDecoderSession;
import io.github.inference4j.generation.GenerationEngine;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.tokenizer.TokenDecoder;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.TokenizerProvider;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Shared builder logic for encoder-decoder model wrappers.
 *
 * <p>Handles model resolution, ONNX session creation, config.json parsing,
 * tokenizer loading, and {@link GenerationEngine} wiring. Subclasses provide
 * only the default {@link TokenizerProvider} and a factory method to create
 * the concrete wrapper type.
 *
 * @param <T> the wrapper type being built
 * @param <B> the concrete builder type (for fluent chaining)
 */
abstract class AbstractEncoderDecoderBuilder<T, B extends AbstractEncoderDecoderBuilder<T, B>> {

    ModelSource modelSource;
    String modelId;
    SessionConfigurer sessionConfigurer;
    TokenizerProvider tokenizerProvider;
    int maxNewTokens = 256;
    float temperature = 0f;
    int topK = 0;
    float topP = 0f;
    final Set<Integer> eosTokenIds = new LinkedHashSet<>();
    final List<String> addedTokens = new ArrayList<>();
    final List<String> extraFiles = new ArrayList<>();

    @SuppressWarnings("unchecked")
    private B self() {
        return (B) this;
    }

    protected abstract TokenizerProvider defaultTokenizerProvider();

    protected abstract T createWrapper(GenerationEngine engine);

    public B modelSource(ModelSource modelSource) {
        this.modelSource = modelSource;
        return self();
    }

    public B modelId(String modelId) {
        this.modelId = modelId;
        return self();
    }

    public B sessionOptions(SessionConfigurer sessionConfigurer) {
        this.sessionConfigurer = sessionConfigurer;
        return self();
    }

    public B tokenizerProvider(TokenizerProvider tokenizerProvider) {
        this.tokenizerProvider = tokenizerProvider;
        return self();
    }

    public B maxNewTokens(int maxNewTokens) {
        this.maxNewTokens = maxNewTokens;
        return self();
    }

    public B temperature(float temperature) {
        this.temperature = temperature;
        return self();
    }

    public B topK(int topK) {
        this.topK = topK;
        return self();
    }

    public B topP(float topP) {
        this.topP = topP;
        return self();
    }

    public B eosTokenId(int eosTokenId) {
        this.eosTokenIds.add(eosTokenId);
        return self();
    }

    public B addedToken(String token) {
        this.addedTokens.add(token);
        return self();
    }

    B requiredFile(String filename) {
        this.extraFiles.add(filename);
        return self();
    }

    public T build() {
        if (modelId == null && modelSource == null) {
            throw new ModelLoadException(
                    "Model source not configured. Set modelId() for HuggingFace models "
                    + "or modelSource() for local models.");
        }
        ModelSource source = modelSource != null
                ? modelSource : HuggingFaceModelSource.defaultInstance();
        String id = modelId != null ? modelId : "";
        TokenizerProvider provider = tokenizerProvider != null
                ? tokenizerProvider : defaultTokenizerProvider();

        List<String> files = new ArrayList<>(List.of(
                "encoder_model.onnx",
                "decoder_model.onnx",
                "decoder_model_with_past.onnx",
                "config.json"));
        files.addAll(provider.requiredFiles());
        files.addAll(extraFiles);

        Path dir = source.resolve(id, files);
        return loadFromDirectory(dir, provider);
    }

    private T loadFromDirectory(Path dir, TokenizerProvider provider) {
        if (!Files.isDirectory(dir)) {
            throw new ModelSourceException("Model directory not found: " + dir);
        }

        Path encoderPath = dir.resolve("encoder_model.onnx");
        Path decoderPath = dir.resolve("decoder_model.onnx");
        Path decoderWithPastPath = dir.resolve("decoder_model_with_past.onnx");
        Path configPath = dir.resolve("config.json");

        if (!Files.exists(encoderPath)) {
            throw new ModelSourceException("Encoder model not found: " + encoderPath);
        }
        if (!Files.exists(decoderPath)) {
            throw new ModelSourceException("Decoder model not found: " + decoderPath);
        }
        if (!Files.exists(decoderWithPastPath)) {
            throw new ModelSourceException(
                    "Decoder-with-past model not found: " + decoderWithPastPath);
        }
        if (!Files.exists(configPath)) {
            throw new ModelSourceException("Config file not found: " + configPath);
        }

        InferenceSession encoderSession = null;
        InferenceSession decoderSession = null;
        InferenceSession decoderWithPastSession = null;

        try {
            encoderSession = createSession(encoderPath);
            decoderSession = createSession(decoderPath);
            decoderWithPastSession = createSession(decoderWithPastPath);

            ModelConfig config = readModelConfig(configPath);

            EncoderDecoderSession generativeSession = new EncoderDecoderSession(
                    encoderSession, decoderSession, decoderWithPastSession,
                    config.decoderStartTokenId());

            TokenizerProvider.TokenizerAndDecoder td = provider.create(dir, addedTokens);
            Tokenizer tokenizer = td.tokenizer();
            TokenDecoder decoder = td.decoder();

            Set<Integer> eos = this.eosTokenIds.isEmpty()
                    ? config.eosTokenIds()
                    : this.eosTokenIds;

            GenerationEngine.Builder engineBuilder = GenerationEngine.builder()
                    .session(generativeSession)
                    .tokenizer(tokenizer)
                    .decoder(decoder)
                    .maxNewTokens(this.maxNewTokens)
                    .temperature(this.temperature)
                    .topK(this.topK)
                    .topP(this.topP);

            for (int eosId : eos) {
                engineBuilder.eosTokenId(eosId);
            }

            return createWrapper(engineBuilder.build());
        } catch (Exception e) {
            closeQuietly(encoderSession);
            closeQuietly(decoderSession);
            closeQuietly(decoderWithPastSession);
            if (e instanceof RuntimeException re) {
                throw re;
            }
            throw new ModelLoadException(
                    "Failed to initialize model: " + e.getMessage(), e);
        }
    }

    private record ModelConfig(int decoderStartTokenId, Set<Integer> eosTokenIds) {}

    private static ModelConfig readModelConfig(Path configPath) {
        try (InputStream is = Files.newInputStream(configPath)) {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(is);

            int decoderStartTokenId = extractDecoderStartTokenId(root);
            Set<Integer> eosTokenIds = extractEosTokenIds(root);

            return new ModelConfig(decoderStartTokenId, eosTokenIds);
        } catch (IOException e) {
            throw new ModelLoadException(
                    "Failed to read config.json: " + e.getMessage(), e);
        }
    }

    private static int extractDecoderStartTokenId(JsonNode root) {
        JsonNode decoderStartNode = root.get("decoder_start_token_id");
        if (decoderStartNode != null && decoderStartNode.isInt()) {
            return decoderStartNode.intValue();
        }

        JsonNode forcedBosNode = root.get("forced_bos_token_id");
        if (forcedBosNode != null && forcedBosNode.isInt()) {
            return forcedBosNode.intValue();
        }

        throw new ModelLoadException(
                "config.json missing decoder_start_token_id — "
                + "set it explicitly or ensure config.json contains this field");
    }

    private static Set<Integer> extractEosTokenIds(JsonNode root) {
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
    }

    private InferenceSession createSession(Path modelPath) {
        return sessionConfigurer != null
                ? InferenceSession.create(modelPath, sessionConfigurer)
                : InferenceSession.create(modelPath);
    }

    static void closeQuietly(AutoCloseable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception ignored) {
                // best-effort cleanup
            }
        }
    }
}
