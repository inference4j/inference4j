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

package io.github.inference4j.audio;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class Wav2Vec2Test {

    // Test vocabulary: 0=<pad>(blank), 1=a, 2=b, 3=c, 4=|
    private static final Vocabulary VOCAB = Vocabulary.of(Map.of(
            0, "<pad>",
            1, "a",
            2, "b",
            3, "c",
            4, "|"
    ));

    private static final int VOCAB_SIZE = 5;
    private static final int BLANK = 0;
    private static final String WORD_DELIM = "|";

    @Test
    void postProcess_decodesWordWithRepeatsAndBlanks() {
        // Sequence: a,a,_,b,b,_,c → "abc"
        float[] logits = buildLogits(1, 1, 0, 2, 2, 0, 3);

        Transcription result = Wav2Vec2.postProcess(logits, 7, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertEquals("abc", result.text());
    }

    @Test
    void postProcess_wordDelimiterBecomesSpace() {
        // Sequence: a,_,|,_,b → "a b"
        float[] logits = buildLogits(1, 0, 4, 0, 2);

        Transcription result = Wav2Vec2.postProcess(logits, 5, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertEquals("a b", result.text());
    }

    @Test
    void postProcess_allBlanks_emptyText() {
        float[] logits = buildLogits(0, 0, 0, 0);

        Transcription result = Wav2Vec2.postProcess(logits, 4, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertEquals("", result.text());
    }

    @Test
    void postProcess_collapsesRepeatedTokens() {
        // a,a,a → "a"
        float[] logits = buildLogits(1, 1, 1);

        Transcription result = Wav2Vec2.postProcess(logits, 3, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertEquals("a", result.text());
    }

    @Test
    void postProcess_sameTokenSeparatedByBlank_repeatedChar() {
        // a,_,a → "aa"
        float[] logits = buildLogits(1, 0, 1);

        Transcription result = Wav2Vec2.postProcess(logits, 3, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertEquals("aa", result.text());
    }

    @Test
    void postProcess_stripsLeadingAndTrailingSpaces() {
        // |,a,b,| → " ab " → stripped to "ab"
        float[] logits = buildLogits(4, 1, 2, 4);

        Transcription result = Wav2Vec2.postProcess(logits, 4, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertEquals("ab", result.text());
    }

    // --- Builder validation ---

    @Test
    void builder_missingSession_throws() {
        assertThrows(IllegalStateException.class, () ->
                Wav2Vec2.builder()
                        .vocabulary(VOCAB)
                        .build());
    }

    @Test
    void builder_missingVocabulary_throws() {
        InferenceSession session = mock(InferenceSession.class);
        assertThrows(IllegalStateException.class, () ->
                Wav2Vec2.builder()
                        .session(session)
                        .build());
    }

    @Test
    void builder_inputNameDefaultsFromSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input_values"));

        Wav2Vec2 model = Wav2Vec2.builder()
                .session(session)
                .vocabulary(VOCAB)
                .build();

        assertNotNull(model);
        verify(session).inputNames();
    }

    // --- Inference flow ---

    @Test
    void transcribe_audioData_returnsTranscription() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input"));

        // CTC output: timesteps produce tokens [a, _, b, _, c] → "abc"
        float[] logits = buildLogits(1, 0, 2, 0, 3);
        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(logits, new long[]{1, 5, VOCAB_SIZE})));

        Wav2Vec2 model = Wav2Vec2.builder()
                .session(session)
                .vocabulary(VOCAB)
                .build();

        Transcription result = model.transcribe(new float[]{0.1f, 0.2f, 0.3f}, 16000);

        assertEquals("abc", result.text());
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input"));

        Wav2Vec2 model = Wav2Vec2.builder()
                .session(session)
                .vocabulary(VOCAB)
                .build();

        model.close();

        verify(session).close();
    }

    /**
     * Builds synthetic CTC logits where the argmax at each timestep is the given token.
     * Sets the target token to 10.0 and all others to -10.0.
     */
    private static float[] buildLogits(int... tokens) {
        float[] logits = new float[tokens.length * VOCAB_SIZE];
        Arrays.fill(logits, -10.0f);
        for (int t = 0; t < tokens.length; t++) {
            logits[t * VOCAB_SIZE + tokens[t]] = 10.0f;
        }
        return logits;
    }
}
