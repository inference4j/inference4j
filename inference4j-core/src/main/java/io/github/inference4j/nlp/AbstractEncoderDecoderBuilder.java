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
import io.github.inference4j.InferenceSession;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.generation.EncoderDecoderSession;
import io.github.inference4j.generation.GenerationEngine;
import io.github.inference4j.model.AbstractModelBuilder;
import io.github.inference4j.model.ModelFiles;
import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.tokenizer.TokenDecoder;
import io.github.inference4j.tokenizer.Tokenizer;
import io.github.inference4j.tokenizer.TokenizerProvider;

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
abstract class AbstractEncoderDecoderBuilder<T, B extends AbstractEncoderDecoderBuilder<T, B>>
        extends AbstractModelBuilder<T, B> {

    TokenizerProvider tokenizerProvider;
    int maxNewTokens = 256;
    float temperature = 0f;
    int topK = 0;
    float topP = 0f;
    final Set<Integer> eosTokenIds = new LinkedHashSet<>();
    final List<String> addedTokens = new ArrayList<>();
    final List<String> extraFiles = new ArrayList<>();
    Integer maxInputLength;
    TruncationPolicy truncation;
    Integer forcedBosTokenId;

    protected abstract TokenizerProvider defaultTokenizerProvider();

    protected abstract T createWrapper(GenerationEngine engine);

    /**
     * Whether the model expects its input to start with the {@code bos_token_id} from {@code config.json}
     * (BART's {@code <s>}), which the tokenizer does not add. Defaults to {@code false}.
     */
    protected boolean prependBosToInput() {
        return false;
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

    /**
     * Maximum input length in tokens. Defaults to the model's limit from {@code config.json}
     * ({@code max_position_embeddings} or {@code n_positions}; 512 for T5).
     */
    public B maxInputLength(int maxInputLength) {
        this.maxInputLength = maxInputLength;
        return self();
    }

    /**
     * What to do with input longer than {@link #maxInputLength(int)}. Defaults to
     * {@link TruncationPolicy#TRUNCATE}, which keeps the first tokens and logs a warning.
     */
    public B truncation(TruncationPolicy truncation) {
        this.truncation = truncation;
        return self();
    }

    /**
     * Token forced as the first generated token, as Hugging Face's {@code forced_bos_token_id} does.
     * Defaults to {@code forced_bos_token_id} from {@code config.json}, if present.
     */
    public B forcedBosTokenId(int forcedBosTokenId) {
        this.forcedBosTokenId = forcedBosTokenId;
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
        String id = modelId != null ? modelId : "";
        TokenizerProvider provider = tokenizerProvider != null
                ? tokenizerProvider : defaultTokenizerProvider();

        List<String> unchecked = new ArrayList<>(provider.requiredFiles());
        unchecked.addAll(extraFiles);
        return load(id, List.of("encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx",
                "config.json"), unchecked, files -> assemble(files, provider));
    }

    private T assemble(ModelFiles files, TokenizerProvider provider) {
        InferenceSession encoderSession = files.session("encoder_model.onnx");
        InferenceSession decoderSession = files.session("decoder_model.onnx");
        InferenceSession decoderWithPastSession = files.session("decoder_with_past_model.onnx");

        JsonNode config = files.json("config.json");
        int decoderStartTokenId = extractDecoderStartTokenId(config);
        // Read even when eosTokenId() is set: a config.json without eos_token_id is rejected either way
        Set<Integer> configEos = EosTokenIds.fromConfig(config);

        EncoderDecoderSession generativeSession = new EncoderDecoderSession(
                encoderSession, decoderSession, decoderWithPastSession, decoderStartTokenId);

        TokenizerProvider.TokenizerAndDecoder td = provider.create(files.directory(), addedTokens);
        Tokenizer tokenizer = td.tokenizer();
        TokenDecoder decoder = td.decoder();

        Set<Integer> eos = this.eosTokenIds.isEmpty() ? configEos : this.eosTokenIds;

        GenerationEngine.Builder engineBuilder = GenerationEngine.builder()
                .session(generativeSession)
                .tokenizer(tokenizer)
                .decoder(decoder)
                .appendEosToInput(true)
                .maxInputLength(maxInputLength != null
                        ? maxInputLength
                        : ModelInputLimits.maxPositions(config).orElse(Integer.MAX_VALUE))
                .truncationGuard(new TruncationGuard(taskName(), truncation))
                .maxNewTokens(this.maxNewTokens)
                .temperature(this.temperature)
                .topK(this.topK)
                .topP(this.topP);

        for (int eosId : eos) {
            engineBuilder.eosTokenId(eosId);
        }
        if (prependBosToInput()) {
            Integer bos = optionalInt(config, "bos_token_id");
            if (bos == null) {
                throw new ModelLoadException("config.json missing bos_token_id, needed to start the input");
            }
            engineBuilder.prependToInput(bos);
        }
        Integer forcedBos = forcedBosTokenId != null ? forcedBosTokenId : optionalInt(config, "forced_bos_token_id");
        if (forcedBos != null) {
            engineBuilder.forcedBosTokenId(forcedBos);
        }
        ModelInputLimits.decoderPositions(config).ifPresent(engineBuilder::maxDecoderPositions);

        return createWrapper(engineBuilder.build());
    }

    private String taskName() {
        Class<?> wrapper = getClass().getEnclosingClass();
        return wrapper != null ? wrapper.getSimpleName() : getClass().getSimpleName();
    }

    private static Integer optionalInt(JsonNode root, String field) {
        JsonNode node = root.get(field);
        return node != null && node.isInt() ? node.intValue() : null;
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
}
