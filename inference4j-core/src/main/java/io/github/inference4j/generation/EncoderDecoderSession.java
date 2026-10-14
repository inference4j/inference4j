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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.NativeTensors;
import io.github.inference4j.Tensor;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * A {@link GenerativeSession} for encoder-decoder (seq2seq) models exported
 * via HuggingFace Optimum as three ONNX files: {@code encoder_model.onnx},
 * {@code decoder_model.onnx}, and {@code decoder_with_past_model.onnx}.
 *
 * <p>The encoder runs once during {@link #prefill(long[])} to produce hidden states.
 * The first decoder step uses {@code decoder_model.onnx} (no past KV cache),
 * and subsequent steps use {@code decoder_with_past_model.onnx} with cached
 * key/value tensors to avoid redundant computation.
 *
 * <p>Two types of KV cache are maintained:
 * <ul>
 *   <li><b>Self-attention cache</b> ({@code past_key_values.N.decoder.key/value}) —
 *       grows with each decode step as new tokens are generated.</li>
 *   <li><b>Cross-attention cache</b> ({@code past_key_values.N.encoder.key/value}) —
 *       frozen after the first decode step, since encoder hidden states don't change.</li>
 * </ul>
 *
 * <p>Both caches, and the encoder hidden states, stay in ONNX Runtime memory between runs;
 * only the logits of each step are copied into Java.
 */
public class EncoderDecoderSession implements GenerativeSession {

    private final InferenceSession encoderSession;
    private final InferenceSession decoderSession;
    private final InferenceSession decoderWithPastSession;
    private final int decoderStartTokenId;
    private final Set<String> selfAttentionOutputNames;
    private final Set<String> crossAttentionOutputNames;
    private final Set<String> decodeOutputNames;

    // The first decoder step's outputs: they hold the cross-attention cache for the whole generation,
    // and the self-attention cache until the first decode step replaces it.
    private NativeTensors firstStep = NativeTensors.empty();
    private NativeTensors crossAttentionCache = NativeTensors.empty();
    private NativeTensors selfAttentionCache = NativeTensors.empty();
    private Tensor encoderAttentionMask;
    private int sequenceLength;

    /**
     * Creates a new encoder-decoder session.
     *
     * @param encoderSession        session for {@code encoder_model.onnx}
     * @param decoderSession        session for {@code decoder_model.onnx} (first decode step)
     * @param decoderWithPastSession session for {@code decoder_with_past_model.onnx} (subsequent steps)
     * @param decoderStartTokenId   the token ID used to start decoder generation
     *                              (e.g., {@code </s>} or {@code <pad>})
     */
    public EncoderDecoderSession(InferenceSession encoderSession,
                                 InferenceSession decoderSession,
                                 InferenceSession decoderWithPastSession,
                                 int decoderStartTokenId) {
        this.encoderSession = encoderSession;
        this.decoderSession = decoderSession;
        this.decoderWithPastSession = decoderWithPastSession;
        this.decoderStartTokenId = decoderStartTokenId;
        int numLayers = (int) decoderWithPastSession.inputNames().stream()
                .filter(n -> n.startsWith("past_key_values.") && n.endsWith(".decoder.key"))
                .count();
        Set<String> selfNames = new HashSet<>();
        Set<String> crossNames = new HashSet<>();
        for (int i = 0; i < numLayers; i++) {
            selfNames.add("present." + i + ".decoder.key");
            selfNames.add("present." + i + ".decoder.value");
            crossNames.add("present." + i + ".encoder.key");
            crossNames.add("present." + i + ".encoder.value");
        }
        this.selfAttentionOutputNames = Set.copyOf(selfNames);
        this.crossAttentionOutputNames = Set.copyOf(crossNames);
        selfNames.add("logits");
        this.decodeOutputNames = Set.copyOf(selfNames);
    }

    @Override
    public ForwardResult prefill(long[] tokenIds) {
        releaseCaches();
        int srcLen = tokenIds.length;

        Map<String, Tensor> encoderInputs = new LinkedHashMap<>();
        encoderInputs.put("input_ids", Tensor.fromLongs(tokenIds, new long[]{1, srcLen}));
        this.encoderAttentionMask = Tensor.fromLongs(ones(srcLen), new long[]{1, srcLen});
        encoderInputs.put("attention_mask", this.encoderAttentionMask);

        Map<String, Tensor> decoderInputs = new LinkedHashMap<>();
        decoderInputs.put("input_ids",
                Tensor.fromLongs(new long[]{decoderStartTokenId}, new long[]{1, 1}));
        decoderInputs.put("encoder_attention_mask", this.encoderAttentionMask);

        NativeTensors step;
        try (NativeTensors encoderOutputs = encoderSession.runNative(
                encoderInputs, Set.of("last_hidden_state"), NativeTensors.empty())) {
            NativeTensors hiddenStates = encoderOutputs.rename(name -> "encoder_hidden_states");
            step = decoderSession.runNative(decoderInputs, null, hiddenStates);
        }

        float[] logits = lastLogits(step);
        this.firstStep = step;
        this.crossAttentionCache = step.select(crossAttentionOutputNames).rename(EncoderDecoderSession::pastName);
        this.selfAttentionCache = step.select(selfAttentionOutputNames).rename(EncoderDecoderSession::pastName);
        this.sequenceLength = 1;

        return new ForwardResult(logits);
    }

    @Override
    public ForwardResult decode(long tokenId) {
        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(new long[]{tokenId}, new long[]{1, 1}));
        inputs.put("encoder_attention_mask", this.encoderAttentionMask);

        NativeTensors step = decoderWithPastSession.runNative(
                inputs, decodeOutputNames, crossAttentionCache, selfAttentionCache);
        float[] logits = lastLogits(step);

        NativeTensors previous = this.selfAttentionCache;
        this.selfAttentionCache = step.select(selfAttentionOutputNames).rename(EncoderDecoderSession::pastName);
        // The first step's self-attention cache shares its memory with the cross-attention cache,
        // which must outlive it; it is freed with the cross-attention cache instead.
        if (sequenceLength > 1) {
            previous.close();
        }
        this.sequenceLength++;

        return new ForwardResult(logits);
    }

    @Override
    public int cacheSequenceLength() {
        return sequenceLength;
    }

    @Override
    public void resetCache() {
        releaseCaches();
    }

    @Override
    public void close() {
        releaseCaches();
        encoderSession.close();
        decoderSession.close();
        decoderWithPastSession.close();
    }

    private void releaseCaches() {
        selfAttentionCache.close();
        firstStep.close();
        selfAttentionCache = NativeTensors.empty();
        crossAttentionCache = NativeTensors.empty();
        firstStep = NativeTensors.empty();
        encoderAttentionMask = null;
        sequenceLength = 0;
    }

    /** The logits of the step's last position: the only output copied into Java. */
    private static float[] lastLogits(NativeTensors step) {
        return step.copy("logits").slice(0, 0).slice(0, -1).toFloats();
    }

    /** A {@code present.*} output's name as the next step's {@code past_key_values.*} input. */
    private static String pastName(String presentName) {
        return presentName.replace("present.", "past_key_values.");
    }

    private long[] ones(int length) {
        long[] result = new long[length];
        Arrays.fill(result, 1L);
        return result;
    }
}
