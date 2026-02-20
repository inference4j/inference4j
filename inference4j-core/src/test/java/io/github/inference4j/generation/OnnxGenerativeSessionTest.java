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

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyMap;
import static org.mockito.Mockito.*;

class OnnxGenerativeSessionTest {

    private InferenceSession inferenceSession;
    private OnnxGenerativeSession session;

    @BeforeEach
    void setUp() {
        inferenceSession = mock(InferenceSession.class);

        // Simulate a 2-layer model with 4 heads, dim 8
        when(inferenceSession.inputShape("past_key_values.0.key"))
                .thenReturn(new long[]{1, 4, 0, 8});
        when(inferenceSession.inputNames())
                .thenReturn(Set.of(
                        "input_ids", "attention_mask", "position_ids",
                        "past_key_values.0.key", "past_key_values.0.value",
                        "past_key_values.1.key", "past_key_values.1.value"
                ));

        session = new OnnxGenerativeSession(inferenceSession);
    }

    @Test
    void cacheSequenceLength_initiallyZero() {
        assertEquals(0, session.cacheSequenceLength());
    }

    @Test
    void cacheSequenceLength_updatesAfterPrefill() {
        stubRunForPrefill();

        session.prefill(new long[]{1, 2, 3});

        assertEquals(3, session.cacheSequenceLength());
    }

    @Test
    void resetCache_resetsSequenceLength() {
        stubRunForPrefill();
        session.prefill(new long[]{1, 2, 3});

        session.resetCache();

        assertEquals(0, session.cacheSequenceLength());
    }

    @Test
    void close_delegatesToInferenceSession() throws Exception {
        session.close();

        verify(inferenceSession).close();
    }

    private void stubRunForPrefill() {
        Map<String, Tensor> outputs = new LinkedHashMap<>();
        // logits tensor: [1, seqLen, vocabSize] — slice(0,0) → [seqLen, vocabSize], slice(0,-1) → [vocabSize]
        outputs.put("logits", Tensor.fromFloats(new float[]{1.0f, 2.0f, 3.0f}, new long[]{1, 1, 3}));
        outputs.put("present.0.key", Tensor.fromFloats(new float[0], new long[]{1, 4, 0, 8}));
        outputs.put("present.0.value", Tensor.fromFloats(new float[0], new long[]{1, 4, 0, 8}));
        outputs.put("present.1.key", Tensor.fromFloats(new float[0], new long[]{1, 4, 0, 8}));
        outputs.put("present.1.value", Tensor.fromFloats(new float[0], new long[]{1, 4, 0, 8}));
        when(inferenceSession.run(anyMap())).thenReturn(outputs);
    }
}
