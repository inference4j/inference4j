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

package io.github.inference4j.generation;

import io.github.inference4j.sampling.CategoricalSampler;
import io.github.inference4j.sampling.GreedySampler;
import io.github.inference4j.sampling.LogitsProcessor;
import io.github.inference4j.sampling.LogitsProcessors;
import io.github.inference4j.sampling.LogitsSampler;
import io.github.inference4j.tokenizer.TokenDecoder;
import io.github.inference4j.tokenizer.Tokenizer;

import java.time.Duration;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Text-in, text-out convenience class that owns the autoregressive generation loop.
 *
 * <p>Wires together a {@link GenerativeSession} (KV cache + forward pass),
 * a {@link Tokenizer}/{@link TokenDecoder} pair, and a sampling pipeline into a
 * single {@link GenerativeTask} that accepts a prompt string and returns generated text.
 *
 * <p>Example usage:
 * <pre>{@code
 * try (var engine = GenerationEngine.builder()
 *         .session(session)
 *         .tokenizer(bpe)
 *         .decoder(bpe)
 *         .eosTokenId(50256)
 *         .temperature(0.7f)
 *         .topP(0.9f)
 *         .maxNewTokens(100)
 *         .build()) {
 *     engine.generate("Hello world", token -> System.out.print(token));
 * }
 * }</pre>
 */
public class GenerationEngine implements GenerativeTask<String, GenerationResult> {

    private final GenerativeSession session;
    private final Tokenizer tokenizer;
    private final TokenDecoder decoder;
    private final ChatTemplate chatTemplate;
    private final LogitsProcessor logitsProcessor;
    private final LogitsSampler sampler;
    private final Set<Integer> eosTokenIds;
    private final int maxNewTokens;
    private final Set<String> stopSequences;
    private final boolean appendEosToInput;

    private GenerationEngine(Builder builder) {
        this.session = builder.session;
        this.tokenizer = builder.tokenizer;
        this.decoder = builder.decoder;
        this.chatTemplate = builder.chatTemplate;
        this.eosTokenIds = Set.copyOf(builder.eosTokenIds);
        this.maxNewTokens = builder.maxNewTokens;
        this.stopSequences = Set.copyOf(builder.stopSequences);
        this.appendEosToInput = builder.appendEosToInput;
        this.logitsProcessor = builder.buildLogitsProcessor();
        this.sampler = builder.buildSampler();
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public GenerationResult generate(String input) {
        return generate(input, token -> {});
    }

    @Override
    public GenerationResult generate(String input, Consumer<String> tokenListener) {
        long startTime = System.nanoTime();

        String prompt = chatTemplate != null ? chatTemplate.format(input) : input;
        long[] inputIds = tokenizer.encode(prompt).inputIds();
        if (appendEosToInput) {
            int eosId = eosTokenIds.iterator().next();
            inputIds = Arrays.copyOf(inputIds, inputIds.length + 1);
            inputIds[inputIds.length - 1] = eosId;
        }
        int promptTokens = inputIds.length;

        session.resetCache();
        ForwardResult result = session.prefill(inputIds);

        TokenStreamer streamer = new TokenStreamer(stopSequences, tokenListener);
        int generatedTokens = 0;

        for (int i = 0; i < maxNewTokens; i++) {
            float[] processed = logitsProcessor.process(result.logits());
            int tokenId = sampler.sample(processed);

            if (eosTokenIds.contains(tokenId)) {
                break;
            }

            String fragment = decoder.decode(tokenId);
            streamer.accept(fragment);
            generatedTokens++;

            if (streamer.isStopped()) {
                break;
            }

            result = session.decode(tokenId);
        }

        if (!streamer.isStopped()) {
            streamer.flush();
        }

        Duration duration = Duration.ofNanos(System.nanoTime() - startTime);
        return new GenerationResult(streamer.getText(), promptTokens, generatedTokens, duration);
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    public static class Builder {

        private GenerativeSession session;
        private Tokenizer tokenizer;
        private TokenDecoder decoder;
        private ChatTemplate chatTemplate;
        private final Set<Integer> eosTokenIds = new LinkedHashSet<>();
        private int maxNewTokens = 256;
        private final Set<String> stopSequences = new LinkedHashSet<>();

        private boolean appendEosToInput = false;
        private float temperature = 0f;
        private int topK = 0;
        private float topP = 0f;

        public Builder session(GenerativeSession session) {
            this.session = session;
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

        public Builder eosTokenId(int tokenId) {
            this.eosTokenIds.add(tokenId);
            return this;
        }

        public Builder maxNewTokens(int maxNewTokens) {
            this.maxNewTokens = maxNewTokens;
            return this;
        }

        public Builder stopSequence(String stopSequence) {
            this.stopSequences.add(stopSequence);
            return this;
        }

        public Builder appendEosToInput(boolean appendEos) {
            this.appendEosToInput = appendEos;
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

        public GenerationEngine build() {
            Objects.requireNonNull(session, "session is required");
            Objects.requireNonNull(tokenizer, "tokenizer is required");
            Objects.requireNonNull(decoder, "decoder is required");
            if (eosTokenIds.isEmpty()) {
                throw new IllegalStateException("At least one eosTokenId is required");
            }
            return new GenerationEngine(this);
        }

        LogitsProcessor buildLogitsProcessor() {
            LogitsProcessor pipeline = LogitsProcessor.identity();
            if (temperature > 0) {
                pipeline = pipeline.andThen(LogitsProcessors.temperature(temperature));
            }
            if (topK > 0) {
                pipeline = pipeline.andThen(LogitsProcessors.topK(topK));
            }
            if (topP > 0) {
                pipeline = pipeline.andThen(LogitsProcessors.topP(topP));
            }
            return pipeline;
        }

        LogitsSampler buildSampler() {
            if (temperature > 0 || topK > 0 || topP > 0) {
                return new CategoricalSampler();
            }
            return new GreedySampler();
        }
    }
}
