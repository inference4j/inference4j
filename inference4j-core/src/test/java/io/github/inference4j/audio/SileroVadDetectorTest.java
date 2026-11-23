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
import java.util.List;

import static org.assertj.core.api.Assertions.*;

class SileroVadDetectorTest {

    private static final int SAMPLE_RATE = 16000;
    private static final int WINDOW_SIZE = 512;
    private static final float THRESHOLD = 0.5f;
    private static final float MIN_SPEECH_DURATION = 0.25f;
    private static final float MIN_SILENCE_DURATION = 0.1f;

    @Test
    void voiceSegment_duration() {
        VoiceSegment segment = new VoiceSegment(1.0f, 3.5f, 0.8f);
        assertThat(segment.duration()).isCloseTo(2.5f, within(0.001f));
    }

    @Test
    void voiceSegment_recordProperties() {
        VoiceSegment segment = new VoiceSegment(0.5f, 2.0f, 0.95f);
        assertThat(segment.start()).isEqualTo(0.5f);
        assertThat(segment.end()).isEqualTo(2.0f);
        assertThat(segment.confidence()).isEqualTo(0.95f);
    }

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                io.github.inference4j.audio.SileroVadDetector.builder()
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_customThreshold_acceptsValues() {
        // Validates builder pattern accepts all values without error
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                SileroVadDetector.builder()
                        .threshold(0.7f)
                        .minSpeechDuration(0.5f)
                        .minSilenceDuration(0.2f)
                        .sampleRate(8000)
                        .windowSizeSamples(256)
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    // --- Inference flow ---
    // The fixture model scores a window as its last sample + 0.01 x the window index (counted by the state it
    // threads through) + 0.001 x the first context sample carried over from the previous window.

    @TempDir
    Path dir;

    @Test
    void detect_audioData_returnsSegments() {
        float[] audioData = new float[2048]; // 4 frames of 512 samples, all scoring above 0.9
        Arrays.fill(audioData, 0.9f);

        try (SileroVadDetector vad = SileroVadDetector.builder()
                .modelSource(TestModels.model(dir, "silero-vad"))
                .threshold(0.5f)
                .minSpeechDuration(0.1f)
                .minSilenceDuration(0.1f)
                .build()) {
            List<VoiceSegment> segments = vad.detect(audioData, 16000);

            assertThat(segments).singleElement().satisfies(segment -> {
                assertThat(segment.start()).isZero();
                assertThat(segment.end()).isCloseTo(2048f / 16000, within(1e-6f));
            });
        }
    }

    @Test
    void detect_noSpeech_returnsEmptySegments() {
        try (SileroVadDetector vad = SileroVadDetector.builder()
                .modelSource(TestModels.model(dir, "silero-vad"))
                .threshold(0.5f)
                .minSpeechDuration(0.1f)
                .minSilenceDuration(0.1f)
                .build()) {
            List<VoiceSegment> segments = vad.detect(new float[2048], 16000);

            assertThat(segments).isEmpty();
        }
    }

    @Test
    void probabilities_returnsPerFrameProbabilities() {
        float[] audioData = new float[1024]; // 2 frames at 512 samples per frame
        audioData[511] = 0.3f;
        audioData[1023] = 0.8f;

        try (SileroVadDetector vad = detector()) {
            float[] probs = vad.probabilities(audioData, 16000);

            assertThat(probs).containsExactly(new float[]{0.30f, 0.81f}, within(1e-5f));
        }
    }

    @Test
    void probabilitiesThreadTheStateAndStartFreshOnEveryCall() {
        try (SileroVadDetector vad = detector()) {
            float[] first = vad.probabilities(new float[2048], 16000);
            float[] second = vad.probabilities(new float[2048], 16000);

            assertThat(first).containsExactly(new float[]{0f, 0.01f, 0.02f, 0.03f}, within(1e-6f));
            assertThat(second).containsExactly(first);
        }
    }

    @Test
    void eachWindowIsPrecededByTheLast64SamplesOfThePreviousOne() {
        float[] audioData = new float[1024];
        audioData[0] = 1f;     // would show in the first window if it had no zero context before it
        audioData[448] = 0.5f; // first of the 64 samples carried into the second window

        try (SileroVadDetector vad = detector()) {
            float[] probs = vad.probabilities(audioData, 16000);

            assertThat(probs).containsExactly(new float[]{0f, 0.01f + 0.0005f}, within(1e-6f));
        }
    }

    // --- Close ---

    @Test
    void closedDetectorRejectsProbabilities() {
        SileroVadDetector vad = detector();

        vad.close();

        assertThatThrownBy(() -> vad.probabilities(new float[512], 16000)).isInstanceOf(IllegalStateException.class);
    }

    private SileroVadDetector detector() {
        return SileroVadDetector.builder().modelSource(TestModels.model(dir, "silero-vad")).build();
    }

    // --- speech padding ---

    @Test
    void zeroPaddingLeavesSegmentsUnchanged() {
        List<VoiceSegment> segments = List.of(new VoiceSegment(1.0f, 2.0f, 0.9f));

        assertThat(SileroVadDetector.pad(segments, 0f, 10f)).isEqualTo(segments);
    }

    @Test
    void paddingExtendsBothSidesAndKeepsConfidence() {
        List<VoiceSegment> padded = SileroVadDetector.pad(
                List.of(new VoiceSegment(1.0f, 2.0f, 0.9f)), 0.2f, 10f);

        assertThat(padded).hasSize(1);
        assertThat(padded.get(0).start()).isCloseTo(0.8f, within(1e-6f));
        assertThat(padded.get(0).end()).isCloseTo(2.2f, within(1e-6f));
        assertThat(padded.get(0).confidence()).isEqualTo(0.9f);
    }

    @Test
    void paddingIsClampedToTheAudio() {
        List<VoiceSegment> padded = SileroVadDetector.pad(
                List.of(new VoiceSegment(0.1f, 2.9f, 0.9f)), 0.5f, 3.0f);

        assertThat(padded.get(0).start()).isZero();
        assertThat(padded.get(0).end()).isEqualTo(3.0f);
    }

    @Test
    void closeNeighboursMeetAtTheMidpointWithoutOverlapping() {
        // gap of 0.16s between segments, padding 0.2s on each side would overlap
        List<VoiceSegment> padded = SileroVadDetector.pad(List.of(
                new VoiceSegment(3.30f, 3.81f, 0.9f),
                new VoiceSegment(3.97f, 4.38f, 0.9f)), 0.2f, 10f);

        float midpoint = (3.81f + 3.97f) / 2;
        assertThat(padded.get(0).end()).isCloseTo(midpoint, within(1e-6f));
        assertThat(padded.get(1).start()).isCloseTo(midpoint, within(1e-6f));
        assertThat(padded.get(0).end()).isLessThanOrEqualTo(padded.get(1).start());
    }

    @Test
    void distantNeighboursGetFullPadding() {
        List<VoiceSegment> padded = SileroVadDetector.pad(List.of(
                new VoiceSegment(1.0f, 2.0f, 0.9f),
                new VoiceSegment(5.0f, 6.0f, 0.9f)), 0.2f, 10f);

        assertThat(padded.get(0).end()).isCloseTo(2.2f, within(1e-6f));
        assertThat(padded.get(1).start()).isCloseTo(4.8f, within(1e-6f));
    }

    @Test
    void negativePaddingIsRejected() {
        assertThatThrownBy(() -> SileroVadDetector.builder().speechPadding(-0.1f))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
