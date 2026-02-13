package io.github.inference4j.audio;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

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
