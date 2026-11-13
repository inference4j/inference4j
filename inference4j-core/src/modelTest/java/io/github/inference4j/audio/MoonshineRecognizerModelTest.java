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

import io.github.inference4j.preprocessing.audio.AudioLoader;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * Expected texts come from a Python port of Moonshine's C++ runtime that matches the official
 * {@code moonshine-voice} library token for token, run on the same 16 kHz audio with the ONNX Runtime
 * version core uses (1.28). Another runtime version can flip a low-confidence punctuation token.
 *
 * <p>Tiny is checked by word error rate against the true transcript instead: its output has near-ties
 * that flip between machines (CI's runner gives "ask what", a local run "as what"), see #141.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MoonshineRecognizerModelTest {

    private static final String TINY = "inference4j/moonshine-streaming-tiny-en";
    private static final String TRANSCRIPT = "And so my fellow Americans, ask not what your country can do"
            + " for you, ask what you can do for your country.";
    /** Tiny gets 1 or 2 of the 22 words wrong depending on the machine; 0.15 allows 3. */
    private static final double TINY_MAX_WER = 0.15;

    private MoonshineRecognizer small;
    private MoonshineRecognizer tiny;
    private Path speech16k;
    private Path speech44k;
    private float[] samples;

    @BeforeAll
    void setUp() throws IOException {
        small = MoonshineRecognizer.builder().build();
        tiny = MoonshineRecognizer.builder().modelId(TINY).build();
        speech16k = fixture("speech-16k.wav");
        speech44k = fixture("speech.wav");
        samples = AudioLoader.load(speech16k).samples();
    }

    @AfterAll
    void tearDown() {
        if (small != null) small.close();
        if (tiny != null) tiny.close();
    }

    private static Path fixture(String name) throws IOException {
        Path file = Files.createTempFile("moonshine-", ".wav");
        file.toFile().deleteOnExit();
        try (InputStream is = MoonshineRecognizerModelTest.class.getResourceAsStream("/fixtures/" + name)) {
            Files.copy(is, file, StandardCopyOption.REPLACE_EXISTING);
        }
        return file;
    }

    /** The clip twice, one second of silence apart: 23 seconds. */
    private float[] twice() {
        float[] audio = new float[samples.length * 2 + 16000];
        System.arraycopy(samples, 0, audio, 0, samples.length);
        System.arraycopy(samples, 0, audio, samples.length + 16000, samples.length);
        return audio;
    }

    @Test
    void smallMatchesReference() {
        assertThat(small.transcribe(speech16k).text()).isEqualTo(
                "And so my fellow Americans, ask not what your country can do for you,"
                        + " ask what you can do for your country.");
    }

    @Test
    void smallMatchesReferenceOnLongerAudio() {
        assertThat(small.transcribe(twice(), 16000).text()).isEqualTo(
                "And so my fellow Americans, ask not what your country can do for you, ask what you can do"
                        + " for your country. And so my fellow Americans, ask not what your country can do for"
                        + " you, ask what you can do for your country.");
    }

    @Test
    void tinyTranscribesWithinErrorBudget() {
        String text = tiny.transcribe(speech16k).text();
        assertThat(WordErrorRate.of(TRANSCRIPT, text)).as("WER of \"%s\"", text).isLessThanOrEqualTo(TINY_MAX_WER);
    }

    @Test
    void tinyTranscribesLongerAudioWithinErrorBudget() {
        String text = tiny.transcribe(twice(), 16000).text();
        assertThat(WordErrorRate.of(TRANSCRIPT + " " + TRANSCRIPT, text))
                .as("WER of \"%s\"", text).isLessThanOrEqualTo(TINY_MAX_WER);
    }

    @Test
    void resamplesOtherSampleRates() {
        assertThat(small.transcribe(speech44k).text())
                .contains("ask not what your country can do for you");
    }

    @Test
    void rejectsAudioLongerThanThirtySeconds() {
        assertThatThrownBy(() -> small.transcribe(new float[16000 * 31], 16000))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("SegmentedSpeechRecognizer");
    }

    @Test
    void segmentedRecognizerHandlesLongAudio() {
        float[] clip = twice();
        float[] longAudio = new float[clip.length * 2 + 16000];
        System.arraycopy(clip, 0, longAudio, 0, clip.length);
        System.arraycopy(clip, 0, longAudio, clip.length + 16000, clip.length);

        try (SpeechRecognizer recognizer = new SegmentedSpeechRecognizer(
                MoonshineRecognizer.builder().build(), SileroVadDetector.builder().build(),
                Duration.ofSeconds(15))) {
            Transcription result = recognizer.transcribe(longAudio, 16000);
            String text = result.text().toLowerCase();

            assertThat(text.split("what your country can do for you", -1)).hasSize(5);
            List<Transcription.Segment> segments = result.segments();
            assertThat(segments).hasSizeGreaterThan(1);
            assertThat(segments).allSatisfy(segment -> assertThat(segment.text()).isNotBlank());
            assertThat(segments.get(0).startTime()).isZero();
            for (int i = 1; i < segments.size(); i++) {
                assertThat(segments.get(i).startTime()).isEqualTo(segments.get(i - 1).endTime());
            }
            assertThat(segments.get(segments.size() - 1).endTime())
                    .isCloseTo((float) longAudio.length / 16000, within(0.001f));
        }
    }
}
