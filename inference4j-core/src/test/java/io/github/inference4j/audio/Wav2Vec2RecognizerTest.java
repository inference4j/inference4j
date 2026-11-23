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

import io.github.inference4j.model.ModelSource;
import io.github.inference4j.TestModels;
import io.github.inference4j.exception.ModelSourceException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

import static org.assertj.core.api.Assertions.*;

class Wav2Vec2RecognizerTest {

    // Test vocabulary: 0=<pad>(blank), 1=a, 2=b, 3=c, 4=|
    private static final io.github.inference4j.preprocessing.audio.Vocabulary VOCAB = io.github.inference4j.preprocessing.audio.Vocabulary.of(Map.of(
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

        Transcription result = io.github.inference4j.audio.Wav2Vec2Recognizer.postProcess(logits, 7, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertThat(result.text()).isEqualTo("abc");
    }

    @Test
    void postProcess_wordDelimiterBecomesSpace() {
        // Sequence: a,_,|,_,b → "a b"
        float[] logits = buildLogits(1, 0, 4, 0, 2);

        Transcription result = Wav2Vec2Recognizer.postProcess(logits, 5, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertThat(result.text()).isEqualTo("a b");
    }

    @Test
    void postProcess_allBlanks_emptyText() {
        float[] logits = buildLogits(0, 0, 0, 0);

        Transcription result = Wav2Vec2Recognizer.postProcess(logits, 4, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertThat(result.text()).isEqualTo("");
    }

    @Test
    void postProcess_collapsesRepeatedTokens() {
        // a,a,a → "a"
        float[] logits = buildLogits(1, 1, 1);

        Transcription result = Wav2Vec2Recognizer.postProcess(logits, 3, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertThat(result.text()).isEqualTo("a");
    }

    @Test
    void postProcess_sameTokenSeparatedByBlank_repeatedChar() {
        // a,_,a → "aa"
        float[] logits = buildLogits(1, 0, 1);

        Transcription result = Wav2Vec2Recognizer.postProcess(logits, 3, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertThat(result.text()).isEqualTo("aa");
    }

    @Test
    void postProcess_stripsLeadingAndTrailingSpaces() {
        // |,a,b,| → " ab " → stripped to "ab"
        float[] logits = buildLogits(4, 1, 2, 4);

        Transcription result = Wav2Vec2Recognizer.postProcess(logits, 4, VOCAB_SIZE,
                VOCAB, BLANK, WORD_DELIM);

        assertThat(result.text()).isEqualTo("ab");
    }

    // --- Builder validation ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                Wav2Vec2Recognizer.builder()
                        .vocabulary(VOCAB)
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_missingVocabulary_throws(@TempDir Path dir) {
        ModelSource source = TestModels.model(dir, "wav2vec2-abc");

        assertThatThrownBy(() -> Wav2Vec2Recognizer.builder().modelSource(source).build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("vocab.json");
    }

    // --- Inference flow ---
    // The fixture model returns logits whose argmax per step is a, blank, b, blank, c.

    @Test
    void transcribe_audioData_returnsTranscription(@TempDir Path dir) {
        try (Wav2Vec2Recognizer model = Wav2Vec2Recognizer.builder()
                .modelSource(TestModels.model(dir, "wav2vec2-abc"))
                .vocabulary(VOCAB)
                .build()) {
            Transcription result = model.transcribe(new float[]{0.1f, 0.2f, 0.3f}, 16000);

            assertThat(result.text()).isEqualTo("abc");
        }
    }

    @Test
    void vocabularyIsReadFromVocabJson(@TempDir Path dir) {
        ModelSource source = TestModels.model(dir, "wav2vec2-abc");
        TestModels.wav2Vec2Vocabulary(dir);

        try (Wav2Vec2Recognizer model = Wav2Vec2Recognizer.builder().modelSource(source).build()) {
            assertThat(model.transcribe(new float[]{0.1f, 0.2f, 0.3f}, 16000).text()).isEqualTo("abc");
        }
    }

    // --- Close ---

    @Test
    void closedRecognizerRejectsTranscribe(@TempDir Path dir) {
        Wav2Vec2Recognizer model = Wav2Vec2Recognizer.builder()
                .modelSource(TestModels.model(dir, "wav2vec2-abc"))
                .vocabulary(VOCAB)
                .build();

        model.close();

        assertThatThrownBy(() -> model.transcribe(new float[]{0.1f, 0.2f, 0.3f}, 16000))
                .isInstanceOf(IllegalStateException.class);
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
