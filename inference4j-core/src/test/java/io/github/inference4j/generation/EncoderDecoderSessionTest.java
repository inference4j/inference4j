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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.Mockito.*;

class EncoderDecoderSessionTest {

    private static final int NUM_LAYERS = 2;
    private static final int NUM_HEADS = 4;
    private static final int HEAD_DIM = 8;
    private static final int HIDDEN_DIM = NUM_HEADS * HEAD_DIM; // 32
    private static final int VOCAB_SIZE = 5;
    private static final int DECODER_START_TOKEN_ID = 2;

    private InferenceSession encoderSession;
    private InferenceSession decoderSession;
    private InferenceSession decoderWithPastSession;
    private EncoderDecoderSession session;

    @BeforeEach
    void setUp() {
        encoderSession = mock(InferenceSession.class);
        decoderSession = mock(InferenceSession.class);
        decoderWithPastSession = mock(InferenceSession.class);

        // decoderWithPastSession exposes input names that reveal the layer count
        when(decoderWithPastSession.inputNames()).thenReturn(Set.of(
                "input_ids",
                "past_key_values.0.decoder.key", "past_key_values.0.decoder.value",
                "past_key_values.0.encoder.key", "past_key_values.0.encoder.value",
                "past_key_values.1.decoder.key", "past_key_values.1.decoder.value",
                "past_key_values.1.encoder.key", "past_key_values.1.encoder.value"
        ));

        session = new EncoderDecoderSession(
                encoderSession, decoderSession, decoderWithPastSession,
                DECODER_START_TOKEN_ID);
    }

    @Test
    void cacheSequenceLength_initiallyZero() {
        assertEquals(0, session.cacheSequenceLength());
    }

    @Test
    @SuppressWarnings("unchecked")
    void prefill_encodesInput_thenRunsFirstDecodeStep() {
        int srcLen = 3;
        long[] inputTokens = {10, 20, 30};

        stubEncoderRun(srcLen);
        stubDecoderRun(srcLen);

        session.prefill(inputTokens);

        // Encoder should be called with input_ids and attention_mask
        ArgumentCaptor<Map<String, Tensor>> encoderCaptor = ArgumentCaptor.forClass(Map.class);
        verify(encoderSession).run(encoderCaptor.capture());
        Map<String, Tensor> encoderInputs = encoderCaptor.getValue();
        assertTrue(encoderInputs.containsKey("input_ids"), "Encoder should receive input_ids");
        assertTrue(encoderInputs.containsKey("attention_mask"), "Encoder should receive attention_mask");
        assertArrayEquals(new long[]{1, srcLen}, encoderInputs.get("input_ids").shape());
        assertArrayEquals(inputTokens, encoderInputs.get("input_ids").toLongs());

        // Decoder should be called with encoder_hidden_states
        ArgumentCaptor<Map<String, Tensor>> decoderCaptor = ArgumentCaptor.forClass(Map.class);
        verify(decoderSession).run(decoderCaptor.capture());
        Map<String, Tensor> decoderInputs = decoderCaptor.getValue();
        assertTrue(decoderInputs.containsKey("input_ids"), "Decoder should receive input_ids");
        assertTrue(decoderInputs.containsKey("encoder_hidden_states"),
                "Decoder should receive encoder_hidden_states");
        assertTrue(decoderInputs.containsKey("encoder_attention_mask"),
                "Decoder should receive encoder_attention_mask");
        // Decoder input_ids should be [1,1] with the decoder start token
        assertArrayEquals(new long[]{1, 1}, decoderInputs.get("input_ids").shape());
        assertArrayEquals(new long[]{DECODER_START_TOKEN_ID},
                decoderInputs.get("input_ids").toLongs());
    }

    @Test
    void prefill_returnsLogits() {
        int srcLen = 3;
        stubEncoderRun(srcLen);
        stubDecoderRun(srcLen);

        ForwardResult result = session.prefill(new long[]{10, 20, 30});

        assertNotNull(result.logits());
        assertEquals(VOCAB_SIZE, result.logits().length);
    }

    @Test
    @SuppressWarnings("unchecked")
    void prefill_storesCrossAttentionCache() {
        int srcLen = 3;
        stubEncoderRun(srcLen);
        stubDecoderRun(srcLen);

        session.prefill(new long[]{10, 20, 30});

        // Now do a decode step — cross-attention cache should be passed
        stubDecoderWithPastRun(srcLen);
        session.decode(42L);

        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(decoderWithPastSession).run(captor.capture());
        Map<String, Tensor> inputs = captor.getValue();

        // Cross-attention cache keys should be present
        for (int i = 0; i < NUM_LAYERS; i++) {
            assertTrue(inputs.containsKey("past_key_values." + i + ".encoder.key"),
                    "Missing cross-attention key for layer " + i);
            assertTrue(inputs.containsKey("past_key_values." + i + ".encoder.value"),
                    "Missing cross-attention value for layer " + i);
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void decode_passesBothCacheTypes() {
        int srcLen = 2;
        stubEncoderRun(srcLen);
        stubDecoderRun(srcLen);
        session.prefill(new long[]{10, 20});

        stubDecoderWithPastRun(srcLen);
        session.decode(42L);

        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(decoderWithPastSession).run(captor.capture());
        Map<String, Tensor> inputs = captor.getValue();

        // Both self-attention and cross-attention cache entries must be present
        for (int i = 0; i < NUM_LAYERS; i++) {
            assertTrue(inputs.containsKey("past_key_values." + i + ".decoder.key"),
                    "Missing self-attention key for layer " + i);
            assertTrue(inputs.containsKey("past_key_values." + i + ".decoder.value"),
                    "Missing self-attention value for layer " + i);
            assertTrue(inputs.containsKey("past_key_values." + i + ".encoder.key"),
                    "Missing cross-attention key for layer " + i);
            assertTrue(inputs.containsKey("past_key_values." + i + ".encoder.value"),
                    "Missing cross-attention value for layer " + i);
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void decode_freezesCrossAttentionCache() {
        int srcLen = 2;
        stubEncoderRun(srcLen);
        stubDecoderRun(srcLen);
        session.prefill(new long[]{10, 20});

        // First decode
        stubDecoderWithPastRun(srcLen);
        session.decode(42L);

        ArgumentCaptor<Map<String, Tensor>> captor1 = ArgumentCaptor.forClass(Map.class);
        verify(decoderWithPastSession).run(captor1.capture());
        Map<String, Tensor> firstDecodeInputs = captor1.getValue();
        Tensor crossKey0First = firstDecodeInputs.get("past_key_values.0.encoder.key");

        // Second decode
        stubDecoderWithPastRun(srcLen);
        session.decode(43L);

        ArgumentCaptor<Map<String, Tensor>> captor2 = ArgumentCaptor.forClass(Map.class);
        verify(decoderWithPastSession, times(2)).run(captor2.capture());
        Map<String, Tensor> secondDecodeInputs = captor2.getAllValues().get(1);
        Tensor crossKey0Second = secondDecodeInputs.get("past_key_values.0.encoder.key");

        // Cross-attention cache should be the exact same object (frozen, not updated)
        assertSame(crossKey0First, crossKey0Second,
                "Cross-attention cache should be frozen (same object across decode steps)");
    }

    @Test
    void decode_incrementsSequenceLength() {
        int srcLen = 2;
        stubEncoderRun(srcLen);
        stubDecoderRun(srcLen);
        session.prefill(new long[]{10, 20});

        assertEquals(1, session.cacheSequenceLength(), "After prefill, sequence length should be 1");

        stubDecoderWithPastRun(srcLen);
        session.decode(42L);
        assertEquals(2, session.cacheSequenceLength(), "After first decode, sequence length should be 2");

        stubDecoderWithPastRun(srcLen);
        session.decode(43L);
        assertEquals(3, session.cacheSequenceLength(), "After second decode, sequence length should be 3");
    }

    @Test
    void resetCache_clearsAllState() {
        int srcLen = 2;
        stubEncoderRun(srcLen);
        stubDecoderRun(srcLen);
        session.prefill(new long[]{10, 20});

        assertEquals(1, session.cacheSequenceLength());

        session.resetCache();

        assertEquals(0, session.cacheSequenceLength());
    }

    @Test
    void close_closesAllSessions() throws Exception {
        session.close();

        verify(encoderSession).close();
        verify(decoderSession).close();
        verify(decoderWithPastSession).close();
    }

    // --- Stubbing helpers ---

    /**
     * Stubs the encoder session to return a last_hidden_state tensor
     * of shape [1, srcLen, HIDDEN_DIM].
     */
    private void stubEncoderRun(int srcLen) {
        Map<String, Tensor> encoderOutputs = new LinkedHashMap<>();
        float[] hiddenState = new float[srcLen * HIDDEN_DIM];
        for (int i = 0; i < hiddenState.length; i++) {
            hiddenState[i] = 0.1f * i;
        }
        encoderOutputs.put("last_hidden_state",
                Tensor.fromFloats(hiddenState, new long[]{1, srcLen, HIDDEN_DIM}));
        when(encoderSession.run(anyMap())).thenReturn(encoderOutputs);
    }

    /**
     * Stubs the decoder session (without past) to return logits and present KV cache tensors.
     * The present keys use the naming convention: present.N.encoder.key/value
     * and present.N.decoder.key/value
     */
    private void stubDecoderRun(int srcLen) {
        Map<String, Tensor> decoderOutputs = new LinkedHashMap<>();
        // logits: [1, 1, VOCAB_SIZE]
        float[] logits = new float[VOCAB_SIZE];
        for (int i = 0; i < logits.length; i++) {
            logits[i] = 1.0f + i;
        }
        decoderOutputs.put("logits", Tensor.fromFloats(logits, new long[]{1, 1, VOCAB_SIZE}));

        // Present KV caches for each layer
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            // Self-attention (decoder): shape [1, NUM_HEADS, 1, HEAD_DIM] (1 position decoded)
            float[] selfCache = new float[NUM_HEADS * 1 * HEAD_DIM];
            decoderOutputs.put("present." + layer + ".decoder.key",
                    Tensor.fromFloats(selfCache, new long[]{1, NUM_HEADS, 1, HEAD_DIM}));
            decoderOutputs.put("present." + layer + ".decoder.value",
                    Tensor.fromFloats(selfCache, new long[]{1, NUM_HEADS, 1, HEAD_DIM}));

            // Cross-attention (encoder): shape [1, NUM_HEADS, srcLen, HEAD_DIM]
            float[] crossCache = new float[NUM_HEADS * srcLen * HEAD_DIM];
            decoderOutputs.put("present." + layer + ".encoder.key",
                    Tensor.fromFloats(crossCache, new long[]{1, NUM_HEADS, srcLen, HEAD_DIM}));
            decoderOutputs.put("present." + layer + ".encoder.value",
                    Tensor.fromFloats(crossCache, new long[]{1, NUM_HEADS, srcLen, HEAD_DIM}));
        }

        when(decoderSession.run(anyMap())).thenReturn(decoderOutputs);
    }

    /**
     * Stubs the decoder-with-past session to return logits and updated self-attention cache.
     * Cross-attention cache is NOT returned by decoder_with_past (it's frozen).
     */
    private void stubDecoderWithPastRun(int srcLen) {
        Map<String, Tensor> outputs = new LinkedHashMap<>();
        // logits: [1, 1, VOCAB_SIZE]
        float[] logits = new float[VOCAB_SIZE];
        for (int i = 0; i < logits.length; i++) {
            logits[i] = 2.0f + i;
        }
        outputs.put("logits", Tensor.fromFloats(logits, new long[]{1, 1, VOCAB_SIZE}));

        // Updated self-attention cache (grows by 1 position each step)
        int selfSeqLen = session.cacheSequenceLength() + 1;
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            float[] selfCache = new float[NUM_HEADS * selfSeqLen * HEAD_DIM];
            outputs.put("present." + layer + ".decoder.key",
                    Tensor.fromFloats(selfCache, new long[]{1, NUM_HEADS, selfSeqLen, HEAD_DIM}));
            outputs.put("present." + layer + ".decoder.value",
                    Tensor.fromFloats(selfCache, new long[]{1, NUM_HEADS, selfSeqLen, HEAD_DIM}));
        }

        when(decoderWithPastSession.run(anyMap())).thenReturn(outputs);
    }
}
