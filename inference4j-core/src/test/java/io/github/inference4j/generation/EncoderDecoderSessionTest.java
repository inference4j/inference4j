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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.ArgumentMatchers.anySet;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.ArgumentMatchers.startsWith;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EncoderDecoderSessionTest {

    private static final int DECODER_START_TOKEN_ID = 2;
    private static final Set<String> DECODE_OUTPUTS = Set.of("logits",
            "present.0.decoder.key", "present.0.decoder.value",
            "present.1.decoder.key", "present.1.decoder.value");

    private InferenceSession encoderSession;
    private InferenceSession decoderSession;
    private InferenceSession decoderWithPastSession;
    private EncoderDecoderSession session;

    private NativeTensors encoderOutputs;
    private NativeTensors hiddenStates;
    private final List<NativeTensors> firstSteps = new ArrayList<>();
    private final List<Cache> crossCaches = new ArrayList<>();
    private final List<Cache> selfCaches = new ArrayList<>();
    private final List<NativeTensors> decodeSteps = new ArrayList<>();

    @BeforeEach
    void setUp() {
        encoderSession = mock(InferenceSession.class);
        decoderSession = mock(InferenceSession.class);
        decoderWithPastSession = mock(InferenceSession.class);

        // decoderWithPastSession exposes input names that reveal the layer count
        when(decoderWithPastSession.inputNames()).thenReturn(Set.of(
                "input_ids", "encoder_attention_mask",
                "past_key_values.0.decoder.key", "past_key_values.0.decoder.value",
                "past_key_values.0.encoder.key", "past_key_values.0.encoder.value",
                "past_key_values.1.decoder.key", "past_key_values.1.decoder.value",
                "past_key_values.1.encoder.key", "past_key_values.1.encoder.value"));

        encoderOutputs = mock(NativeTensors.class);
        hiddenStates = mock(NativeTensors.class);
        when(encoderOutputs.rename(any())).thenReturn(hiddenStates);
        when(encoderSession.runNative(anyMap(), any(), any(NativeTensors.class))).thenReturn(encoderOutputs);
        when(decoderSession.runNative(anyMap(), any(), any(NativeTensors.class)))
                .thenAnswer(call -> newFirstStep());
        when(decoderWithPastSession.runNative(anyMap(), any(), any(NativeTensors.class), any(NativeTensors.class)))
                .thenAnswer(call -> newDecodeStep());

        session = new EncoderDecoderSession(
                encoderSession, decoderSession, decoderWithPastSession, DECODER_START_TOKEN_ID);
    }

    @Test
    void cacheSequenceLengthIsInitiallyZero() {
        assertThat(session.cacheSequenceLength()).isZero();
    }

    @Test
    @SuppressWarnings("unchecked")
    void prefillEncodesTheInputThenRunsTheFirstDecoderStep() {
        long[] inputTokens = {10, 20, 30};

        session.prefill(inputTokens);

        ArgumentCaptor<Map<String, Tensor>> encoderCaptor = ArgumentCaptor.forClass(Map.class);
        verify(encoderSession).runNative(encoderCaptor.capture(), eq(Set.of("last_hidden_state")),
                same(NativeTensors.empty()));
        Map<String, Tensor> encoderInputs = encoderCaptor.getValue();
        assertThat(encoderInputs.get("input_ids").shape()).isEqualTo(new long[]{1, 3});
        assertThat(encoderInputs.get("input_ids").toLongs()).isEqualTo(inputTokens);
        assertThat(encoderInputs.get("attention_mask").toLongs()).containsExactly(1, 1, 1);

        // The encoder hidden states reach the decoder without leaving ONNX Runtime memory
        ArgumentCaptor<Map<String, Tensor>> decoderCaptor = ArgumentCaptor.forClass(Map.class);
        verify(decoderSession).runNative(decoderCaptor.capture(), isNull(), same(hiddenStates));
        Map<String, Tensor> decoderInputs = decoderCaptor.getValue();
        assertThat(decoderInputs.get("input_ids").shape()).isEqualTo(new long[]{1, 1});
        assertThat(decoderInputs.get("input_ids").toLongs()).containsExactly(DECODER_START_TOKEN_ID);
        assertThat(decoderInputs.get("encoder_attention_mask").shape()).isEqualTo(new long[]{1, 3});
        assertThat(decoderInputs).doesNotContainKey("encoder_hidden_states");
    }

    @Test
    void prefillFreesTheEncoderOutputsAfterTheFirstDecoderStep() {
        session.prefill(new long[]{10, 20, 30});

        verify(encoderOutputs).close();
        verify(firstSteps.get(0), never()).close();
    }

    @Test
    void prefillReturnsTheLogits() {
        ForwardResult result = session.prefill(new long[]{10, 20, 30});

        assertThat(result.logits()).containsExactly(1f, 2f, 3f);
        assertThat(session.cacheSequenceLength()).isEqualTo(1);
    }

    @Test
    void prefillKeepsTheCachesUnderTheirPastNames() {
        session.prefill(new long[]{10, 20});

        ArgumentCaptor<UnaryOperator<String>> naming =
                ArgumentCaptor.forClass(UnaryOperator.class);
        verify(crossCaches.get(0).selected()).rename(naming.capture());
        assertThat(naming.getValue().apply("present.3.encoder.key")).isEqualTo("past_key_values.3.encoder.key");
        verify(firstSteps.get(0)).select(Set.of(
                "present.0.encoder.key", "present.0.encoder.value",
                "present.1.encoder.key", "present.1.encoder.value"));
        verify(firstSteps.get(0)).select(Set.of(
                "present.0.decoder.key", "present.0.decoder.value",
                "present.1.decoder.key", "present.1.decoder.value"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void decodePassesBothCachesAndTheEncoderAttentionMask() {
        session.prefill(new long[]{10, 20, 30});

        session.decode(42L);

        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(decoderWithPastSession).runNative(captor.capture(), eq(DECODE_OUTPUTS),
                same(crossCaches.get(0).kept()), same(selfCaches.get(0).kept()));
        Map<String, Tensor> inputs = captor.getValue();
        assertThat(inputs.get("input_ids").toLongs()).containsExactly(42L);
        assertThat(inputs.get("encoder_attention_mask").shape()).isEqualTo(new long[]{1, 3});
        assertThat(inputs).doesNotContainKeys("past_key_values.0.decoder.key", "past_key_values.0.encoder.key");
    }

    @Test
    void crossAttentionCacheIsReusedForEveryStepAndStaysOpen() {
        session.prefill(new long[]{10, 20});

        session.decode(42L);
        session.decode(43L);

        NativeTensors cross = crossCaches.get(0).kept();
        verify(decoderWithPastSession, times(2)).runNative(anyMap(), anySet(), same(cross), any(NativeTensors.class));
        verify(decoderWithPastSession).runNative(anyMap(), anySet(), same(cross), same(selfCaches.get(1).kept()));
        verify(cross, never()).close();
        verify(firstSteps.get(0), never()).close();
    }

    @Test
    void eachDecodeStepFreesThePreviousSelfAttentionCache() {
        session.prefill(new long[]{10, 20});

        session.decode(42L);
        session.decode(43L);

        // The first step's self-attention cache shares memory with the cross-attention cache
        verify(selfCaches.get(0).kept(), never()).close();
        verify(selfCaches.get(1).kept()).close();
        verify(selfCaches.get(2).kept(), never()).close();
        assertThat(session.cacheSequenceLength()).isEqualTo(3);
    }

    @Test
    void onlyTheLogitsAreCopiedIntoJava() {
        session.prefill(new long[]{10, 20});
        ForwardResult result = session.decode(42L);

        assertThat(result.logits()).containsExactly(4f, 5f, 6f);
        verify(firstSteps.get(0), never()).copy(startsWith("present"));
        verify(decodeSteps.get(0)).copy("logits");
        verify(decodeSteps.get(0), never()).copy(startsWith("present"));
    }

    @Test
    void resetCacheFreesTheCaches() {
        session.prefill(new long[]{10, 20});
        session.decode(42L);

        session.resetCache();

        verify(selfCaches.get(1).kept()).close();
        verify(firstSteps.get(0)).close();
        assertThat(session.cacheSequenceLength()).isZero();
    }

    @Test
    void prefillFreesThePreviousGeneration() {
        session.prefill(new long[]{10, 20});
        session.decode(42L);

        session.prefill(new long[]{30, 40});

        verify(selfCaches.get(1).kept()).close();
        verify(firstSteps.get(0)).close();
        verify(firstSteps.get(1), never()).close();
        assertThat(session.cacheSequenceLength()).isEqualTo(1);
    }

    @Test
    void closeFreesTheCachesAndClosesAllSessions() throws Exception {
        session.prefill(new long[]{10, 20});

        session.close();

        verify(firstSteps.get(0)).close();
        verify(encoderSession).close();
        verify(decoderSession).close();
        verify(decoderWithPastSession).close();
    }

    // --- Stubbing helpers ---

    /** A selected view and the renamed view kept from it. */
    private record Cache(NativeTensors selected, NativeTensors kept) {
    }

    private static Cache newCache() {
        NativeTensors selected = mock(NativeTensors.class);
        NativeTensors kept = mock(NativeTensors.class);
        when(selected.rename(any())).thenReturn(kept);
        return new Cache(selected, kept);
    }

    /** The decoder_model.onnx outputs: logits plus both caches. */
    private NativeTensors newFirstStep() {
        NativeTensors step = mock(NativeTensors.class);
        when(step.copy("logits")).thenReturn(Tensor.fromFloats(new float[]{1f, 2f, 3f}, new long[]{1, 1, 3}));
        Cache cross = newCache();
        Cache self = newCache();
        when(step.select(argThat(names -> names != null && names.contains("present.0.encoder.key"))))
                .thenReturn(cross.selected());
        when(step.select(argThat(names -> names != null && names.contains("present.0.decoder.key"))))
                .thenReturn(self.selected());
        firstSteps.add(step);
        crossCaches.add(cross);
        selfCaches.add(self);
        return step;
    }

    /** The decoder_with_past_model.onnx outputs: logits plus the grown self-attention cache. */
    private NativeTensors newDecodeStep() {
        NativeTensors step = mock(NativeTensors.class);
        when(step.copy("logits")).thenReturn(Tensor.fromFloats(new float[]{4f, 5f, 6f}, new long[]{1, 1, 3}));
        Cache self = newCache();
        when(step.select(anySet())).thenReturn(self.selected());
        decodeSteps.add(step);
        selfCaches.add(self);
        return step;
    }
}
