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

import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SileroVADTest {

    private static final int SAMPLE_RATE = 16000;
    private static final int WINDOW_SIZE = 512;
    private static final float THRESHOLD = 0.5f;
    private static final float MIN_SPEECH_DURATION = 0.25f;
    private static final float MIN_SILENCE_DURATION = 0.1f;

    @Test
    void voiceSegment_duration() {
        VoiceSegment segment = new VoiceSegment(1.0f, 3.5f, 0.8f);
        assertEquals(2.5f, segment.duration(), 0.001f);
    }

    @Test
    void voiceSegment_recordProperties() {
        VoiceSegment segment = new VoiceSegment(0.5f, 2.0f, 0.95f);
        assertEquals(0.5f, segment.start());
        assertEquals(2.0f, segment.end());
        assertEquals(0.95f, segment.confidence());
    }

    @Test
    void extractSegments_singleSpeechSegment() throws Exception {
        SileroVAD vad = createTestVad();

        // Create probabilities with enough speech frames to pass minSpeechDuration
        // minSpeechDuration = 0.25s, at window 512 and 16kHz = ~8 frames minimum
        // We need at least 8 frames of speech, plus silence at start/end
        float[] probabilities = new float[20];
        // Fill with silence
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = 0.2f;
        }
        // Set speech frames (frames 3-14 = 12 frames of speech)
        for (int i = 3; i <= 14; i++) {
            probabilities[i] = 0.8f;
        }
        int totalSamples = probabilities.length * WINDOW_SIZE;

        List<VoiceSegment> segments = invokeExtractSegments(vad, probabilities, totalSamples);

        assertEquals(1, segments.size());
        VoiceSegment segment = segments.get(0);

        // Start time: frame 3 * 512 / 16000 = 0.096s
        assertEquals(0.096f, segment.start(), 0.001f);
        assertTrue(segment.confidence() > 0.7f);
    }

    @Test
    void extractSegments_multipleSpeechSegments() throws Exception {
        SileroVAD vad = createTestVad();

        // Two speech segments with silence in between
        // Segment 1: frames 2-7 (6 frames = ~192ms at window 512)
        // Silence: frames 8-12 (5 frames)
        // Segment 2: frames 13-18 (6 frames)
        float[] probabilities = new float[25];
        // Fill with silence
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = 0.2f;
        }
        // First speech segment
        for (int i = 2; i <= 10; i++) {
            probabilities[i] = 0.8f;
        }
        // Second speech segment
        for (int i = 16; i <= 24; i++) {
            probabilities[i] = 0.75f;
        }

        int totalSamples = probabilities.length * WINDOW_SIZE;
        List<VoiceSegment> segments = invokeExtractSegments(vad, probabilities, totalSamples);

        assertEquals(2, segments.size());
        assertTrue(segments.get(0).start() < segments.get(1).start());
    }

    @Test
    void extractSegments_noSpeech() throws Exception {
        SileroVAD vad = createTestVad();

        // All frames below threshold
        float[] probabilities = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f};
        int totalSamples = probabilities.length * WINDOW_SIZE;

        List<VoiceSegment> segments = invokeExtractSegments(vad, probabilities, totalSamples);

        assertTrue(segments.isEmpty());
    }

    @Test
    void extractSegments_shortSpeechFiltered() throws Exception {
        SileroVAD vad = createTestVad();

        // Only 2 frames of speech - too short (minSpeechDuration = 0.25s = ~8 frames)
        float[] probabilities = {0.2f, 0.8f, 0.8f, 0.2f, 0.2f};
        int totalSamples = probabilities.length * WINDOW_SIZE;

        List<VoiceSegment> segments = invokeExtractSegments(vad, probabilities, totalSamples);

        // Should be filtered out because it's shorter than minSpeechDuration
        assertTrue(segments.isEmpty());
    }

    @Test
    void extractSegments_speechAtEndOfAudio() throws Exception {
        SileroVAD vad = createTestVad();

        // Speech that continues to the end of audio (10 frames of speech)
        float[] probabilities = new float[15];
        for (int i = 0; i < 5; i++) {
            probabilities[i] = 0.2f;
        }
        for (int i = 5; i < 15; i++) {
            probabilities[i] = 0.8f;
        }

        int totalSamples = probabilities.length * WINDOW_SIZE;
        List<VoiceSegment> segments = invokeExtractSegments(vad, probabilities, totalSamples);

        assertEquals(1, segments.size());
        // End time should be at the end of audio
        float expectedEndTime = (float) totalSamples / SAMPLE_RATE;
        assertEquals(expectedEndTime, segments.get(0).end(), 0.01f);
    }

    @Test
    void extractSegments_confidenceCalculation() throws Exception {
        SileroVAD vad = createTestVad();

        // Create speech segment with known confidence values (10 frames)
        float[] probabilities = new float[15];
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = 0.2f;
        }
        // Set speech frames with specific confidence values
        probabilities[2] = 0.6f;
        probabilities[3] = 0.8f;
        probabilities[4] = 0.9f;
        probabilities[5] = 0.7f;
        probabilities[6] = 0.8f;
        probabilities[7] = 0.85f;
        probabilities[8] = 0.75f;
        probabilities[9] = 0.65f;
        probabilities[10] = 0.7f;
        probabilities[11] = 0.8f;

        int totalSamples = probabilities.length * WINDOW_SIZE;
        List<VoiceSegment> segments = invokeExtractSegments(vad, probabilities, totalSamples);

        assertEquals(1, segments.size());
        // Average confidence should be around 0.755
        float expectedConfidence = (0.6f + 0.8f + 0.9f + 0.7f + 0.8f + 0.85f + 0.75f + 0.65f + 0.7f + 0.8f) / 10f;
        assertEquals(expectedConfidence, segments.get(0).confidence(), 0.01f);
    }

    @Test
    void builder_requiresSession() {
        SileroVAD.Builder builder = SileroVAD.builder();

        IllegalStateException ex = assertThrows(IllegalStateException.class, builder::build);
        assertEquals("InferenceSession is required", ex.getMessage());
    }

    @Test
    void builder_customThreshold() {
        // This test validates builder pattern works - actual session would fail
        SileroVAD.Builder builder = SileroVAD.builder()
                .threshold(0.7f)
                .minSpeechDuration(0.5f)
                .minSilenceDuration(0.2f)
                .sampleRate(8000)
                .windowSizeSamples(256);

        // We can't build without a session, but the builder accepts all values
        assertThrows(IllegalStateException.class, builder::build);
    }

    /**
     * Creates a test VAD instance with reflection to bypass session requirement.
     */
    private SileroVAD createTestVad() throws Exception {
        java.lang.reflect.Constructor<SileroVAD> constructor =
                SileroVAD.class.getDeclaredConstructor(
                        io.github.inference4j.InferenceSession.class,
                        int.class, int.class, int.class, float.class, float.class, float.class);
        constructor.setAccessible(true);
        int contextSize = 64; // Default context size for 16kHz
        return constructor.newInstance(null, SAMPLE_RATE, WINDOW_SIZE, contextSize,
                THRESHOLD, MIN_SPEECH_DURATION, MIN_SILENCE_DURATION);
    }

    /**
     * Invokes the package-private extractSegments method via reflection for testing.
     */
    private List<VoiceSegment> invokeExtractSegments(SileroVAD vad, float[] probabilities, int totalSamples)
            throws Exception {
        Method method = SileroVAD.class.getDeclaredMethod("extractSegments", float[].class, int.class);
        method.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<VoiceSegment> result = (List<VoiceSegment>) method.invoke(vad, probabilities, totalSamples);
        return result;
    }
}

