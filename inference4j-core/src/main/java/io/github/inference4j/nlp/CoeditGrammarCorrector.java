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
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;
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

/**
 * Grammar corrector backed by CoEdit encoder-decoder models.
 *
 * <p>CoEdit (by Grammarly) is a T5-based model fine-tuned for text editing tasks.
 * This class uses the prompt prefix {@code "Fix grammatical errors in this sentence: "}
 * as documented by Grammarly for CoEdit grammar correction.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * // CoEdit Base (250M) — fast grammar correction
 * try (var corrector = CoeditGrammarCorrector.coeditBase().build()) {
 *     String fixed = corrector.correct("She don't likes the weathers today");
 *     System.out.println(fixed); // "She doesn't like the weather today"
 * }
 *
 * // CoEdit Large (780M) — best quality grammar correction
 * try (var corrector = CoeditGrammarCorrector.coeditLarge().build()) {
 *     String fixed = corrector.correct("He go to school yesterday");
 *     System.out.println(fixed); // "He went to school yesterday"
 * }
 * }</pre>
 *
 * @see TextGenerator
 * @see GrammarCorrector
 * @see GenerationResult
 */
public class CoeditGrammarCorrector implements TextGenerator, GrammarCorrector {

    private static final String GRAMMAR_PREFIX =
            "Fix grammatical errors in this sentence: ";

    private final GenerationEngine engine;

    CoeditGrammarCorrector(GenerationEngine engine) {
        this.engine = engine;
    }

    /**
     * CoEdit Base (250M parameters) preset.
     *
     * <p>Fast grammar correction. Downloads from {@code inference4j/coedit-base}
     * on first use.
     */
    public static Builder coeditBase() {
        return builder().modelId("inference4j/coedit-base");
    }

    /**
     * CoEdit Large (780M parameters) preset.
     *
     * <p>Best quality grammar correction. Downloads from {@code inference4j/coedit-large}
     * on first use. Requires external data files.
     */
    public static Builder coeditLarge() {
        return builder()
                .modelId("inference4j/coedit-large")
                .requiredFile("decoder_model.onnx_data")
                .requiredFile("encoder_model.onnx_data");
    }

    /**
     * Generic builder for custom CoEdit or compatible encoder-decoder models.
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

    // --- GrammarCorrector ---

    @Override
    public GenerationResult correct(String text, Consumer<String> tokenListener) {
        return engine.generate(GRAMMAR_PREFIX + text, tokenListener);
    }

    @Override
    public void close() throws Exception {
        engine.close();
    }

    public static class Builder {

        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private TokenizerProvider tokenizerProvider;
        private int maxNewTokens = 256;
        private float temperature = 0f;
        private int topK = 0;
        private float topP = 0f;
        private final Set<Integer> eosTokenIds = new LinkedHashSet<>();
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

        public Builder addedToken(String token) {
            this.addedTokens.add(token);
            return this;
        }

        Builder requiredFile(String filename) {
            this.extraFiles.add(filename);
            return this;
        }

        public CoeditGrammarCorrector build() {
            if (modelId == null && modelSource == null) {
                throw new ModelLoadException(
                        "Model source not configured. Set modelId() for HuggingFace models "
                        + "or modelSource() for local models.");
            }
            ModelSource source = modelSource != null
                    ? modelSource : HuggingFaceModelSource.defaultInstance();
            String id = modelId != null ? modelId : "";
            TokenizerProvider provider = tokenizerProvider != null
                    ? tokenizerProvider : SentencePieceBpeTokenizer.provider();

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

        private CoeditGrammarCorrector loadFromDirectory(Path dir,
                                                          TokenizerProvider provider) {
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

                int decoderStartTokenId = readDecoderStartTokenId(configPath);

                EncoderDecoderSession generativeSession = new EncoderDecoderSession(
                        encoderSession, decoderSession, decoderWithPastSession,
                        decoderStartTokenId);

                TokenizerProvider.TokenizerAndDecoder td = provider.create(dir, addedTokens);
                Tokenizer tokenizer = td.tokenizer();
                TokenDecoder decoder = td.decoder();

                Set<Integer> eos = this.eosTokenIds.isEmpty()
                        ? readEosTokenIds(configPath)
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

                return new CoeditGrammarCorrector(engineBuilder.build());
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

        private static int readDecoderStartTokenId(Path configPath) {
            try {
                ObjectMapper mapper = new ObjectMapper();
                JsonNode root = mapper.readTree(Files.newInputStream(configPath));

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
            } catch (IOException e) {
                throw new ModelLoadException(
                        "Failed to read config.json: " + e.getMessage(), e);
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

        private InferenceSession createSession(Path modelPath) {
            return sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
        }

        private static void closeQuietly(AutoCloseable closeable) {
            if (closeable != null) {
                try {
                    closeable.close();
                } catch (Exception ignored) {
                    // best-effort cleanup
                }
            }
        }
    }
}
