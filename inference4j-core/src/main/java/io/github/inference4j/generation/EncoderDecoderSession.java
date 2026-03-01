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
import io.github.inference4j.Tensor;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

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
 */
public class EncoderDecoderSession implements GenerativeSession {

    private final InferenceSession encoderSession;
    private final InferenceSession decoderSession;
    private final InferenceSession decoderWithPastSession;
    private final int decoderStartTokenId;
    private final int numLayers;

    private Map<String, Tensor> decoderSelfAttentionCache;
    private Map<String, Tensor> crossAttentionCache;
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
        this.numLayers = (int) decoderWithPastSession.inputNames().stream()
                .filter(n -> n.startsWith("past_key_values.") && n.endsWith(".decoder.key"))
                .count();
        this.decoderSelfAttentionCache = new LinkedHashMap<>();
        this.crossAttentionCache = new LinkedHashMap<>();
    }

    @Override
    public ForwardResult prefill(long[] tokenIds) {
        int srcLen = tokenIds.length;

        long[] attentionMask = ones(srcLen);
        Map<String, Tensor> encoderInputs = new LinkedHashMap<>();
        encoderInputs.put("input_ids", Tensor.fromLongs(tokenIds, new long[]{1, srcLen}));
        this.encoderAttentionMask = Tensor.fromLongs(attentionMask, new long[]{1, srcLen});
        encoderInputs.put("attention_mask", this.encoderAttentionMask);

        Map<String, Tensor> encoderOutputs = encoderSession.run(encoderInputs);
        Tensor encoderHiddenStates = encoderOutputs.get("last_hidden_state");

        Map<String, Tensor> decoderInputs = new LinkedHashMap<>();
        decoderInputs.put("input_ids",
                Tensor.fromLongs(new long[]{decoderStartTokenId}, new long[]{1, 1}));
        decoderInputs.put("encoder_hidden_states", encoderHiddenStates);
        decoderInputs.put("encoder_attention_mask", this.encoderAttentionMask);

        Map<String, Tensor> decoderOutputs = decoderSession.run(decoderInputs);

        float[] logits = decoderOutputs.get("logits").slice(0, 0).slice(0, -1).toFloats();

        this.crossAttentionCache.clear();
        for (int i = 0; i < numLayers; i++) {
            crossAttentionCache.put("past_key_values." + i + ".encoder.key",
                    decoderOutputs.get("present." + i + ".encoder.key"));
            crossAttentionCache.put("past_key_values." + i + ".encoder.value",
                    decoderOutputs.get("present." + i + ".encoder.value"));
        }

        this.decoderSelfAttentionCache.clear();
        for (int i = 0; i < numLayers; i++) {
            decoderSelfAttentionCache.put("past_key_values." + i + ".decoder.key",
                    decoderOutputs.get("present." + i + ".decoder.key"));
            decoderSelfAttentionCache.put("past_key_values." + i + ".decoder.value",
                    decoderOutputs.get("present." + i + ".decoder.value"));
        }

        this.sequenceLength = 1;

        return new ForwardResult(logits);
    }

    @Override
    public ForwardResult decode(long tokenId) {
        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put("input_ids", Tensor.fromLongs(new long[]{tokenId}, new long[]{1, 1}));

        inputs.put("encoder_attention_mask", this.encoderAttentionMask);

        inputs.putAll(decoderSelfAttentionCache);

        inputs.putAll(crossAttentionCache);

        Map<String, Tensor> outputs = decoderWithPastSession.run(inputs);

        float[] logits = outputs.get("logits").slice(0, 0).slice(0, -1).toFloats();

        for (int i = 0; i < numLayers; i++) {
            decoderSelfAttentionCache.put("past_key_values." + i + ".decoder.key",
                    outputs.get("present." + i + ".decoder.key"));
            decoderSelfAttentionCache.put("past_key_values." + i + ".decoder.value",
                    outputs.get("present." + i + ".decoder.value"));
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
        decoderSelfAttentionCache.clear();
        crossAttentionCache.clear();
        encoderAttentionMask = null;
        sequenceLength = 0;
    }

    @Override
    public void close() throws Exception {
        encoderSession.close();
        decoderSession.close();
        decoderWithPastSession.close();
    }

    private long[] ones(int length) {
        long[] result = new long[length];
        Arrays.fill(result, 1L);
        return result;
    }
}
