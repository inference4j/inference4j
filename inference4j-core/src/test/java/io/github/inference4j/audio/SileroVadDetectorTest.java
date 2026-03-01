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
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

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

    @Test
    void detect_audioData_returnsSegments() {
        InferenceSession session = mock(InferenceSession.class);

        // Mock model to return high speech probability for all frames
        // Shape [1] for output, shape [2, 1, 128] for state
        float[] outputProbs = {0.9f};
        float[] stateN = new float[2 * 1 * 128]; // State tensor dimensions
        when(session.run(any())).thenReturn(Map.of(
                "output", Tensor.fromFloats(outputProbs, new long[]{1}),
                "stateN", Tensor.fromFloats(stateN, new long[]{2, 1, 128})
        ));

        SileroVadDetector vad = SileroVadDetector.builder()
                .session(session)
                .threshold(0.5f)
                .minSpeechDuration(0.1f)
                .minSilenceDuration(0.1f)
                .build();

        // Provide enough samples to generate multiple frames (512 samples per frame at 16kHz)
        float[] audioData = new float[2048]; // ~128ms = 4 frames
        List<VoiceSegment> segments = vad.detect(audioData, 16000);

        // Should detect speech since all frames return 0.9 probability
        assertThat(segments).isNotEmpty();
        verify(session, atLeastOnce()).run(any());
    }

    @Test
    void detect_noSpeech_returnsEmptySegments() {
        InferenceSession session = mock(InferenceSession.class);

        // Mock model to return low speech probability for all frames
        float[] outputProbs = {0.1f};
        float[] stateN = new float[2 * 1 * 128];
        when(session.run(any())).thenReturn(Map.of(
                "output", Tensor.fromFloats(outputProbs, new long[]{1}),
                "stateN", Tensor.fromFloats(stateN, new long[]{2, 1, 128})
        ));

        SileroVadDetector vad = SileroVadDetector.builder()
                .session(session)
                .threshold(0.5f)
                .minSpeechDuration(0.1f)
                .minSilenceDuration(0.1f)
                .build();

        float[] audioData = new float[2048];
        List<VoiceSegment> segments = vad.detect(audioData, 16000);

        // Should not detect speech since all frames are below threshold
        assertThat(segments).isEmpty();
    }

    @Test
    void probabilities_returnsPerFrameProbabilities() {
        InferenceSession session = mock(InferenceSession.class);

        // Return different probabilities per frame
        float[] stateN = new float[2 * 1 * 128];
        when(session.run(any()))
                .thenReturn(Map.of(
                        "output", Tensor.fromFloats(new float[]{0.3f}, new long[]{1}),
                        "stateN", Tensor.fromFloats(stateN, new long[]{2, 1, 128})))
                .thenReturn(Map.of(
                        "output", Tensor.fromFloats(new float[]{0.8f}, new long[]{1}),
                        "stateN", Tensor.fromFloats(stateN, new long[]{2, 1, 128})));

        SileroVadDetector vad = SileroVadDetector.builder()
                .session(session)
                .build();

        float[] audioData = new float[1024]; // 2 frames at 512 samples per frame
        float[] probs = vad.probabilities(audioData, 16000);

        assertThat(probs).hasSize(2);
        assertThat(probs[0]).isCloseTo(0.3f, within(0.001f));
        assertThat(probs[1]).isCloseTo(0.8f, within(0.001f));
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);

        SileroVadDetector vad = SileroVadDetector.builder()
                .session(session)
                .build();

        vad.close();

        verify(session).close();
    }

}
