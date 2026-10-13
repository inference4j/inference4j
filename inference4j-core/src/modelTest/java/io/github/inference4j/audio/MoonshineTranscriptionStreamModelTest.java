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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MoonshineTranscriptionStreamModelTest {

    private static final String PHRASE = "what your country can do for you";

    private MoonshineRecognizer recognizer;
    private SileroVadDetector vad;
    private float[] speech16k;
    private float[] speech44k;

    @BeforeAll
    void setUp() throws IOException {
        recognizer = MoonshineRecognizer.builder().modelId("inference4j/moonshine-streaming-tiny-en").build();
        vad = SileroVadDetector.builder().build();
        speech16k = AudioLoader.load(fixture("speech-16k.wav")).samples();
        speech44k = AudioLoader.load(fixture("speech.wav")).samples();
    }

    @AfterAll
    void tearDown() {
        if (recognizer != null) recognizer.close();
        if (vad != null) vad.close();
    }

    private static Path fixture(String name) throws IOException {
        Path file = Files.createTempFile("moonshine-stream-", ".wav");
        file.toFile().deleteOnExit();
        try (InputStream is = MoonshineTranscriptionStreamModelTest.class.getResourceAsStream("/fixtures/" + name)) {
            Files.copy(is, file, StandardCopyOption.REPLACE_EXISTING);
        }
        return file;
    }

    /** Records every event as "started", "changed" or "completed", with the line it carried. */
    private static final class Recorder implements TranscriptListener {
        final List<String> events = new ArrayList<>();
        final List<TranscriptLine> lines = new ArrayList<>();
        final List<TranscriptLine> completed = new ArrayList<>();

        @Override
        public void onLineStarted(TranscriptLine line) {
            record("started", line);
        }

        @Override
        public void onLineTextChanged(TranscriptLine line) {
            record("changed", line);
        }

        @Override
        public void onLineCompleted(TranscriptLine line) {
            record("completed", line);
            completed.add(line);
        }

        private void record(String event, TranscriptLine line) {
            events.add(event + ":" + line.id());
            lines.add(line);
        }

        String text() {
            return String.join(" ", completed.stream().map(TranscriptLine::text).toList()).toLowerCase();
        }
    }

    private static void feed(TranscriptionStream stream, float[] audio, int sampleRate, int chunk) {
        for (int offset = 0; offset < audio.length; offset += chunk) {
            stream.addAudio(Arrays.copyOfRange(audio, offset, Math.min(audio.length, offset + chunk)), sampleRate);
        }
    }

    private static float[] concat(float[]... parts) {
        int length = Arrays.stream(parts).mapToInt(p -> p.length).sum();
        float[] all = new float[length];
        int offset = 0;
        for (float[] part : parts) {
            System.arraycopy(part, 0, all, offset, part.length);
            offset += part.length;
        }
        return all;
    }

    @Test
    void transcribesSpeechAsItArrives() {
        Recorder recorder = new Recorder();
        float[] audio = concat(new float[16000], speech16k, new float[32000], speech16k, new float[16000]);

        try (TranscriptionStream stream = recognizer.stream(vad).listener(recorder).build()) {
            feed(stream, audio, 16000, 1600);
        }

        // The clip has natural pauses, so each copy is several lines
        assertThat(recorder.completed).hasSizeGreaterThanOrEqualTo(2);
        assertThat(recorder.text().split(PHRASE, -1)).hasSize(3);
        assertThat(recorder.events.stream().filter(e -> e.startsWith("changed"))).isNotEmpty();
    }

    @Test
    void eventsComeInOrderForEachLine() {
        Recorder recorder = new Recorder();
        float[] audio = concat(new float[16000], speech16k, new float[32000], speech16k, new float[16000]);

        try (TranscriptionStream stream = recognizer.stream(vad).listener(recorder).build()) {
            feed(stream, audio, 16000, 1600);
        }

        long open = -1;
        long expectedId = 0;
        for (String event : recorder.events) {
            String type = event.substring(0, event.indexOf(':'));
            long id = Long.parseLong(event.substring(event.indexOf(':') + 1));
            switch (type) {
                case "started" -> {
                    assertThat(open).as("only one line open at a time").isEqualTo(-1);
                    assertThat(id).isEqualTo(expectedId++);
                    open = id;
                }
                case "changed" -> assertThat(id).isEqualTo(open);
                case "completed" -> {
                    assertThat(id).isEqualTo(open);
                    open = -1;
                }
                default -> throw new AssertionError(event);
            }
        }
        assertThat(open).isEqualTo(-1);
        for (TranscriptLine line : recorder.lines) {
            assertThat(line.end()).isGreaterThan(line.start());
            assertThat(line.completed()).isEqualTo(recorder.completed.contains(line));
        }
    }

    @Test
    void linesAreTimedOnTheStreamsTimeline() {
        Recorder recorder = new Recorder();
        float[] audio = concat(new float[16000 * 3], speech16k, new float[16000]);

        try (TranscriptionStream stream = recognizer.stream(vad).listener(recorder).build()) {
            feed(stream, audio, 16000, 1600);
        }

        TranscriptLine first = recorder.completed.get(0);
        // Speech begins at 3 s; the line starts up to 0.5 s earlier, with the audio kept from before it
        assertThat(first.start()).isBetween(Duration.ofMillis(2400), Duration.ofMillis(3300));
    }

    @Test
    void resamplesOtherSampleRates() {
        Recorder recorder = new Recorder();
        float[] audio = concat(new float[44100], speech44k, new float[44100]);

        try (TranscriptionStream stream = recognizer.stream(vad).listener(recorder).build()) {
            feed(stream, audio, 44100, 4410);
        }

        assertThat(recorder.text()).contains(PHRASE);
    }

    @Test
    void longSpeechWithoutPausesIsCutAtTheMaximum() {
        Recorder recorder = new Recorder();
        float[] audio = concat(speech16k, speech16k, speech16k);

        try (TranscriptionStream stream = recognizer.stream(vad)
                .maxLineDuration(Duration.ofSeconds(8))
                .listener(recorder)
                .build()) {
            feed(stream, audio, 16000, 1600);
        }

        assertThat(recorder.completed).hasSizeGreaterThanOrEqualTo(4);
        for (TranscriptLine line : recorder.completed) {
            assertThat(line.end().minus(line.start())).isLessThanOrEqualTo(Duration.ofMillis(8100));
        }
    }

    @Test
    void flushCompletesTheOpenLine() {
        Recorder recorder = new Recorder();

        try (TranscriptionStream stream = recognizer.stream(vad).listener(recorder).build()) {
            // The clip's first pause comes after 2.5 s
            feed(stream, Arrays.copyOf(speech16k, 16000 * 2), 16000, 1600);
            assertThat(recorder.completed).isEmpty();

            stream.flush();

            assertThat(recorder.completed).hasSize(1);
            assertThat(recorder.completed.get(0).text()).isNotBlank();
        }
        assertThat(recorder.completed).hasSize(1);
    }

    @Test
    void closeCompletesTheOpenLineAndEndsTheStream() {
        Recorder recorder = new Recorder();
        TranscriptionStream stream = recognizer.stream(vad).listener(recorder).build();
        feed(stream, Arrays.copyOf(speech16k, 16000 * 2), 16000, 1600);

        stream.close();

        assertThat(recorder.completed).hasSize(1);
        assertThatThrownBy(() -> stream.addAudio(new float[1600], 16000)).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void silenceProducesNoLines() {
        Recorder recorder = new Recorder();

        try (TranscriptionStream stream = recognizer.stream(vad).listener(recorder).build()) {
            feed(stream, new float[16000 * 5], 16000, 1600);
        }

        assertThat(recorder.events).isEmpty();
    }

    @Test
    void sampleRateMustNotChange() {
        try (TranscriptionStream stream = recognizer.stream(vad).build()) {
            stream.addAudio(new float[1600], 16000);

            assertThatThrownBy(() -> stream.addAudio(new float[4410], 44100))
                    .isInstanceOf(IllegalArgumentException.class);
        }
    }

    @Test
    void builderValidatesItsOptions() {
        assertThatThrownBy(() -> recognizer.stream(vad).updateInterval(Duration.ZERO))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> recognizer.stream(vad).maxLineDuration(Duration.ofSeconds(31)))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> recognizer.stream(vad).maxLineDuration(Duration.ofMillis(500)))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void closingTheStreamLeavesTheRecognizerOpen() {
        try (TranscriptionStream stream = recognizer.stream(vad).build()) {
            feed(stream, speech16k, 16000, 1600);
        }

        assertThat(recognizer.transcribe(speech16k, 16000).text().toLowerCase()).contains(PHRASE);
    }
}
