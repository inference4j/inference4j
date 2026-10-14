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

import io.github.inference4j.processing.TruncationGuard;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.sampling.CategoricalSampler;
import io.github.inference4j.sampling.GreedySampler;
import io.github.inference4j.sampling.LogitsProcessor;
import io.github.inference4j.sampling.LogitsProcessors;
import io.github.inference4j.sampling.LogitsSampler;
import io.github.inference4j.tokenizer.EncodedInput;
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
    private final Integer inputPrefixTokenId;
    private final Integer forcedBosTokenId;
    private final int maxInputLength;
    private final TruncationGuard truncationGuard;
    private final boolean keepPromptEnd;

    /** Stands in for the user message when splitting a chat template into prefix and suffix. */
    private static final String MESSAGE_MARKER = "\uE000";

    private GenerationEngine(Builder builder) {
        this.session = builder.session;
        this.tokenizer = builder.tokenizer;
        this.decoder = builder.decoder;
        this.chatTemplate = builder.chatTemplate;
        this.eosTokenIds = Set.copyOf(builder.eosTokenIds);
        this.maxNewTokens = builder.maxNewTokens;
        this.stopSequences = Set.copyOf(builder.stopSequences);
        this.appendEosToInput = builder.appendEosToInput;
        this.inputPrefixTokenId = builder.inputPrefixTokenId;
        this.forcedBosTokenId = builder.forcedBosTokenId;
        this.maxInputLength = builder.maxInputLength;
        this.keepPromptEnd = builder.keepPromptEnd;
        this.truncationGuard = builder.truncationGuard != null
                ? builder.truncationGuard
                : new TruncationGuard("GenerationEngine", TruncationPolicy.TRUNCATE);
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
        // Leave room for the tokens added around the prompt below
        int addedTokens = (appendEosToInput ? 1 : 0) + (inputPrefixTokenId != null ? 1 : 0);
        int tokenLimit = maxInputLength != Integer.MAX_VALUE
                ? maxInputLength - addedTokens : maxInputLength;
        EncodedInput encoded = tokenizer.encode(prompt, tokenLimit);
        long[] inputIds;
        if (keepPromptEnd && encoded.truncated()) {
            truncationGuard.check(encoded, tokenLimit, "kept the chat template and the end of the message");
            inputIds = keepTemplateAndEnd(input, prompt, tokenLimit);
        } else {
            truncationGuard.check(encoded, tokenLimit);
            inputIds = encoded.inputIds();
        }
        if (appendEosToInput) {
            int eosId = eosTokenIds.iterator().next();
            inputIds = Arrays.copyOf(inputIds, inputIds.length + 1);
            inputIds[inputIds.length - 1] = eosId;
        }
        if (inputPrefixTokenId != null) {
            long[] prefixed = new long[inputIds.length + 1];
            prefixed[0] = inputPrefixTokenId;
            System.arraycopy(inputIds, 0, prefixed, 1, inputIds.length);
            inputIds = prefixed;
        }
        int promptTokens = inputIds.length;

        session.resetCache();
        ForwardResult result = session.prefill(inputIds);
        if (forcedBosTokenId != null && maxNewTokens > 0) {
            // The first generated token is fixed; it is fed to the model but is not part of the text
            result = session.decode(forcedBosTokenId);
        }

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

    /**
     * Fits an over-long prompt into {@code limit} tokens for a decoder-only model, which answers only
     * after the chat template's closing tokens and continues from the end of its input. Keeps the
     * template's prefix (including any system prompt), its suffix (the cue that the assistant should
     * answer), and as much of the <em>end</em> of the message as fits; tokens the tokenizer adds on its
     * own, such as BOS, are kept too. Without a template, or when the template's tokens do not line up
     * with the full prompt's, it keeps those added tokens and the end of the prompt.
     */
    private long[] keepTemplateAndEnd(String input, String prompt, int limit) {
        long[] full = tokenizer.encode(prompt, Integer.MAX_VALUE).inputIds();
        long[] added = tokenizer.encode("", Integer.MAX_VALUE).inputIds();
        int leading = 0;                                   // added at the start, e.g. BOS
        while (leading < added.length && leading < full.length && added[leading] == full[leading]) {
            leading++;
        }
        int trailing = added.length - leading;             // added at the end, e.g. EOS

        int prefixLength = leading;
        int suffixLength = trailing;
        String[] parts = templateParts(input, prompt);
        if (parts != null) {
            long[] prefix = tokenizer.encode(parts[0], Integer.MAX_VALUE).inputIds();
            long[] suffix = tokenizer.encode(parts[1], Integer.MAX_VALUE).inputIds();
            int p = prefix.length - trailing;
            int s = suffix.length - leading;
            boolean aligned = p >= 0 && s >= 0 && p + s < full.length
                    && Arrays.equals(full, 0, p, prefix, 0, p)
                    && Arrays.equals(full, full.length - s, full.length, suffix, leading, suffix.length);
            if (aligned && p + s < limit) {
                prefixLength = p;
                suffixLength = s;
            }
        }
        if (prefixLength + suffixLength >= limit) {
            return Arrays.copyOfRange(full, full.length - limit, full.length);
        }

        int messageTokens = limit - prefixLength - suffixLength;
        long[] kept = new long[limit];
        System.arraycopy(full, 0, kept, 0, prefixLength);
        System.arraycopy(full, full.length - suffixLength - messageTokens, kept, prefixLength, messageTokens);
        System.arraycopy(full, full.length - suffixLength, kept, prefixLength + messageTokens, suffixLength);
        return kept;
    }

    /**
     * The chat template's text before and after the message, or {@code null} when there is no template or
     * it does not simply wrap the message.
     */
    private String[] templateParts(String input, String prompt) {
        if (chatTemplate == null) {
            return null;
        }
        String marked = chatTemplate.format(MESSAGE_MARKER);
        int at = marked.indexOf(MESSAGE_MARKER);
        if (at < 0 || marked.indexOf(MESSAGE_MARKER, at + 1) >= 0) {
            return null;
        }
        String prefix = marked.substring(0, at);
        String suffix = marked.substring(at + MESSAGE_MARKER.length());
        return prompt.equals(prefix + input + suffix) ? new String[]{prefix, suffix} : null;
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
        private Integer inputPrefixTokenId;
        private Integer forcedBosTokenId;
        private int maxInputLength = Integer.MAX_VALUE;
        private boolean keepPromptEnd;
        private TruncationGuard truncationGuard;
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

        /**
         * A token placed before the prompt, such as BART's {@code <s>}, for tokenizers that do not add it
         * themselves. Counts toward {@link #maxInputLength(int)}.
         */
        public Builder prependToInput(int tokenId) {
            this.inputPrefixTokenId = tokenId;
            return this;
        }

        /**
         * A token forced as the first generated token, after the decoder start token, as Hugging Face's
         * {@code forced_bos_token_id} does (BART's CNN summarizers force {@code <s>}). It is fed to the model
         * but is neither streamed nor part of the result text, and does not count toward
         * {@link #maxNewTokens(int)}.
         */
        public Builder forcedBosTokenId(int tokenId) {
            this.forcedBosTokenId = tokenId;
            return this;
        }

        /**
         * Maximum prompt length in tokens, including any token added before or after it. Longer prompts are
         * handled by the {@link #truncationGuard(TruncationGuard) truncation guard}. Defaults to
         * no limit.
         */
        public Builder maxInputLength(int maxInputLength) {
            this.maxInputLength = maxInputLength;
            return this;
        }

        /**
         * How an over-long prompt is truncated. {@code false} (the default) keeps its beginning, which is
         * right for an encoder's input. {@code true} keeps the chat template and the end of the message,
         * which is what a decoder-only model needs: it answers after the template's closing tokens and
         * continues from the end of its input.
         */
        public Builder keepPromptEnd(boolean keepPromptEnd) {
            this.keepPromptEnd = keepPromptEnd;
            return this;
        }

        /** Policy for prompts longer than {@link #maxInputLength(int)}. Defaults to truncating with a warning. */
        public Builder truncationGuard(TruncationGuard truncationGuard) {
            this.truncationGuard = truncationGuard;
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
