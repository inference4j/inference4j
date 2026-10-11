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

import io.github.inference4j.preprocessing.audio.AudioData;
import io.github.inference4j.preprocessing.audio.AudioWriter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class SegmentedSpeechRecognizerTest {

    // A low sample rate keeps the test arrays small: one second is 100 samples.
    private static final int RATE = 100;

    private final RecordingRecognizer recognizer = new RecordingRecognizer();
    private final VoiceActivityDetector vad = mock(VoiceActivityDetector.class);
    private final SegmentedSpeechRecognizer segmented = new SegmentedSpeechRecognizer(recognizer, vad);

    @Test
    void passesAudioThatFitsThroughInOnePiece() {
        Transcription result = segmented.transcribe(seconds(20), RATE);

        assertThat(recognizer.pieceSeconds).containsExactly(20f);
        assertThat(result.text()).isEqualTo("piece-1");
        verify(vad, never()).detect(any(float[].class), anyInt());
    }

    @Test
    void cutsAtTheLatestPauseBeforeTheLimit() {
        // Speech with pauses centred at 12.5 s, 28.5 s, 45.5 s and 59 s
        speech(0, 12, 13, 28, 29, 45, 46, 58, 60, 70);

        segmented.transcribe(seconds(70), RATE);

        // 0 → 28.5 (latest pause within 30 s), 28.5 → 45.5 (59 is past 58.5), then the 24.5 s that fit
        assertThat(recognizer.pieceSeconds).containsExactly(28.5f, 17f, 24.5f);
    }

    @Test
    void cutsAtTheLimitWhenSpeechHasNoPause() {
        speech(0, 70);

        segmented.transcribe(seconds(70), RATE);

        assertThat(recognizer.pieceSeconds).containsExactly(30f, 30f, 10f);
    }

    @Test
    void ignoresAPauseRightAtTheStartOfAPiece() {
        // The pause at 30.5 s is less than a second into the second piece, so it is skipped
        speech(0, 30, 31, 70);

        segmented.transcribe(seconds(70), RATE);

        assertThat(recognizer.pieceSeconds).containsExactly(30f, 30f, 10f);
    }

    @Test
    void honoursACustomMaximumSegmentLength() {
        speech(0, 9, 10, 19, 20, 29);
        var tenSeconds = new SegmentedSpeechRecognizer(recognizer, vad, Duration.ofSeconds(10));

        tenSeconds.transcribe(seconds(29), RATE);

        assertThat(recognizer.pieceSeconds).containsExactly(9.5f, 10f, 9.5f);
    }

    @Test
    void joinsPieceTextsAndSkipsBlankOnes() {
        speech(0, 70);
        recognizer.texts.addAll(List.of(" Marie Curie moved to Paris. ", "  ", "She won."));

        Transcription result = segmented.transcribe(seconds(70), RATE);

        assertThat(result.text()).isEqualTo("Marie Curie moved to Paris. She won.");
    }

    @Test
    void shiftsSegmentsToTheRecordingTimeline() {
        speech(0, 70);
        recognizer.segment = new Transcription.Segment("word", 1f, 2f);

        Transcription result = segmented.transcribe(seconds(70), RATE);

        assertThat(result.segments()).containsExactly(
                new Transcription.Segment("word", 1f, 2f),
                new Transcription.Segment("word", 31f, 32f),
                new Transcription.Segment("word", 61f, 62f));
    }

    @Test
    void loadsAudioFromAPath(@TempDir Path dir) throws IOException {
        Path wav = dir.resolve("speech.wav");
        AudioWriter.write(new AudioData(seconds(5), 16000), wav);

        segmented.transcribe(wav);

        assertThat(recognizer.pieceSeconds).hasSize(1);
        assertThat(recognizer.lastSampleRate).isEqualTo(16000);
    }

    @Test
    void closesTheRecognizerAndTheDetector() {
        segmented.close();

        assertThat(recognizer.closed).isTrue();
        verify(vad).close();
    }

    @Test
    void rejectsAMaximumSegmentOfOneSecondOrLess() {
        assertThatThrownBy(() -> new SegmentedSpeechRecognizer(recognizer, vad, Duration.ofSeconds(1)))
                .isInstanceOf(IllegalArgumentException.class);
    }

    /** Speech segments given as start/end pairs in seconds. */
    private void speech(float... bounds) {
        List<VoiceSegment> segments = new ArrayList<>();
        for (int i = 0; i < bounds.length; i += 2) {
            segments.add(new VoiceSegment(bounds[i], bounds[i + 1], 0.9f));
        }
        when(vad.detect(any(float[].class), anyInt())).thenReturn(segments);
    }

    private static float[] seconds(float seconds) {
        return new float[Math.round(seconds * RATE)];
    }

    /** Records the length of every piece it receives and answers with queued or numbered texts. */
    private static class RecordingRecognizer implements SpeechRecognizer {
        final List<Float> pieceSeconds = new ArrayList<>();
        final Deque<String> texts = new ArrayDeque<>();
        Transcription.Segment segment;
        int lastSampleRate;
        boolean closed;

        @Override
        public Transcription transcribe(Path audioPath) {
            throw new UnsupportedOperationException("the segmented recognizer passes samples");
        }

        @Override
        public Transcription transcribe(float[] audioData, int sampleRate) {
            pieceSeconds.add((float) audioData.length / sampleRate);
            lastSampleRate = sampleRate;
            String text = texts.isEmpty() ? "piece-" + pieceSeconds.size() : texts.poll();
            return new Transcription(text, segment == null ? List.of() : List.of(segment));
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
