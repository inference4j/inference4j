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
package io.github.inference4j.genai.audio;

import io.github.inference4j.audio.SegmentedSpeechRecognizer;
import io.github.inference4j.audio.SileroVadDetector;
import io.github.inference4j.audio.SpeechRecognizer;
import io.github.inference4j.audio.Transcription;
import io.github.inference4j.preprocessing.audio.AudioData;
import io.github.inference4j.preprocessing.audio.AudioLoader;
import io.github.inference4j.preprocessing.audio.AudioWriter;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration test that transcribes real speech with Whisper tiny.
 *
 * <p>The model is downloaded from HuggingFace on first run (~225 MB) through the default
 * model source, so this also covers resolving a Whisper model on a cold cache.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class WhisperSpeechModelModelTest {

    private static final String EXPECTED = "And so my fellow Americans ask not what your country can do for you "
            + "ask what you can do for your country.";

    private WhisperSpeechModel whisper;
    private Path speechFixture;

    @BeforeAll
    void setUp() throws IOException {
        whisper = WhisperSpeechModel.builder()
                .modelId("inference4j/whisper-tiny-genai")
                .build();
        speechFixture = Files.createTempFile("speech-fixture-", ".wav");
        speechFixture.toFile().deleteOnExit();
        try (InputStream is = WhisperSpeechModelModelTest.class.getResourceAsStream("/fixtures/speech.wav")) {
            Files.copy(is, speechFixture, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    @AfterAll
    void tearDown() {
        if (whisper != null) {
            whisper.close();
        }
    }

    @Test
    void transcribesSpeechFixture() {
        Transcription result = whisper.transcribe(speechFixture);

        assertThat(result.text()).isEqualTo(EXPECTED);
    }

    @Test
    void worksAsSpeechRecognizer() {
        SpeechRecognizer recognizer = whisper;

        assertThat(recognizer.run(speechFixture).text()).isEqualTo(EXPECTED);
    }

    @Test
    void transcribesRawSamplesLikeTheFile() {
        AudioData audio = AudioLoader.load(speechFixture);

        Transcription result = whisper.transcribe(audio.samples(), audio.sampleRate());

        assertThat(result.text()).isEqualTo(EXPECTED);
    }

    @Test
    void transcribesAudioLongerThan30Seconds() throws IOException {
        Path longAudio = repeatFixture(4);
        try {
            Transcription result = whisper.transcribe(longAudio);

            // Four repetitions across two 30-second chunks: every one must be transcribed. A cut
            // that lands mid-sentence can add a garbled partial repeat (see issue #86), so this
            // checks coverage rather than exact text.
            assertThat(countOccurrences(result.text(), "what you can do for your country"))
                    .isGreaterThanOrEqualTo(4);
        } finally {
            Files.deleteIfExists(longAudio);
        }
    }

    @Test
    void segmentedRecognizerCutsLongAudioAtPauses() throws IOException {
        Path longAudio = repeatFixture(4);
        // A separate instance: closing the wrapper closes the model it wraps
        try (SpeechRecognizer recognizer = new SegmentedSpeechRecognizer(
                WhisperSpeechModel.builder().modelId("inference4j/whisper-tiny-genai").build(),
                SileroVadDetector.builder().build())) {

            Transcription result = recognizer.transcribe(longAudio);

            // Cut in the half-second gaps, every repetition comes out whole and exactly once
            assertThat(countOccurrences(result.text(), "what you can do for your country")).isEqualTo(4);
        } finally {
            Files.deleteIfExists(longAudio);
        }
    }

    /** The speech fixture repeated with half-second gaps, written to a temporary WAV. */
    private Path repeatFixture(int times) throws IOException {
        AudioData audio = AudioLoader.load(speechFixture);
        int gap = audio.sampleRate() / 2;
        float[] samples = new float[audio.samples().length * times + gap * (times - 1)];
        for (int i = 0; i < times; i++) {
            System.arraycopy(audio.samples(), 0, samples, i * (audio.samples().length + gap), audio.samples().length);
        }
        Path path = Files.createTempFile("speech-long-", ".wav");
        AudioWriter.write(new AudioData(samples, audio.sampleRate()), path);
        return path;
    }

    private static int countOccurrences(String text, String phrase) {
        int count = 0;
        for (int i = text.indexOf(phrase); i >= 0; i = text.indexOf(phrase, i + phrase.length())) {
            count++;
        }
        return count;
    }
}
