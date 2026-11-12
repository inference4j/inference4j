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

package io.github.inference4j.langchain4j;

import dev.langchain4j.data.document.Document;
import io.github.inference4j.audio.SpeechRecognizer;
import io.github.inference4j.audio.Transcription;
import io.github.inference4j.audio.VoiceActivityDetector;
import io.github.inference4j.audio.VoiceSegment;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.preprocessing.audio.AudioData;
import io.github.inference4j.preprocessing.audio.AudioWriter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class Inference4jAudioParserTest {

    private static final int SAMPLE_RATE = 16_000;

    @TempDir
    Path tempDir;

    private Path wav;

    @BeforeEach
    void writeTwoSecondWav() throws IOException {
        float[] samples = new float[SAMPLE_RATE * 2];
        for (int i = 0; i < samples.length; i++) {
            samples[i] = (float) (0.5 * Math.sin(2 * Math.PI * 440 * i / SAMPLE_RATE));
        }
        wav = tempDir.resolve("input.wav");
        AudioWriter.write(new AudioData(samples, SAMPLE_RATE), wav);
    }

    // --- parse() without VAD ---

    @Test
    void parse_withoutVad_returnsTranscriptAsSingleDocument() throws IOException {
        StubRecognizer recognizer = new StubRecognizer("  hello world  ");

        Document document = parse(new Inference4jAudioParser(recognizer));

        assertThat(document.text()).isEqualTo("hello world");
        assertThat(recognizer.pathCalls).isEqualTo(1);
        assertThat(recognizer.sliceLengths).isEmpty();
    }

    @Test
    void parse_handsRecognizerAFileWithTheStreamContents() throws IOException {
        StubRecognizer recognizer = new StubRecognizer("hello");

        parse(new Inference4jAudioParser(recognizer));

        assertThat(recognizer.bytesSeen).isEqualTo(Files.size(wav));
    }

    @Test
    void parse_deletesTempFileOnSuccess() throws IOException {
        StubRecognizer recognizer = new StubRecognizer("hello");

        parse(new Inference4jAudioParser(recognizer));

        assertThat(recognizer.lastPath).isNotNull().doesNotExist();
    }

    @Test
    void parse_deletesTempFileWhenRecognizerThrows() {
        StubRecognizer recognizer = new StubRecognizer("hello");
        recognizer.failWith = new InferenceException("model exploded");

        assertThatThrownBy(() -> parse(new Inference4jAudioParser(recognizer)))
                .isInstanceOf(InferenceException.class)
                .hasMessage("model exploded");
        assertThat(recognizer.lastPath).isNotNull().doesNotExist();
    }

    @Test
    void parse_blankTranscript_throwsNoSpeech() {
        StubRecognizer recognizer = new StubRecognizer("   ");

        assertThatThrownBy(() -> parse(new Inference4jAudioParser(recognizer)))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("No speech detected");
    }

    // --- parse() with VAD ---

    @Test
    void parse_withVad_transcribesEachSegmentSliceAndJoins() throws IOException {
        StubRecognizer recognizer = new StubRecognizer("first", "second");
        StubVad vad = new StubVad(
                new VoiceSegment(0.0f, 0.5f, 0.9f),
                new VoiceSegment(1.0f, 1.25f, 0.8f));

        Document document = parse(new Inference4jAudioParser(recognizer, vad));

        assertThat(document.text()).isEqualTo("first second");
        assertThat(document.metadata().getInteger(Inference4jAudioParser.VAD_SEGMENT_COUNT)).isEqualTo(2);
        assertThat(recognizer.sliceLengths).containsExactly(SAMPLE_RATE / 2, SAMPLE_RATE / 4);
        assertThat(recognizer.pathCalls).isZero();
        assertThat(vad.sampleRateSeen).isEqualTo(SAMPLE_RATE);
    }

    @Test
    void parse_withVad_clampsSegmentRunningPastEndOfAudio() throws IOException {
        StubRecognizer recognizer = new StubRecognizer("tail");
        StubVad vad = new StubVad(new VoiceSegment(1.5f, 5.0f, 0.9f));

        parse(new Inference4jAudioParser(recognizer, vad));

        assertThat(recognizer.sliceLengths).containsExactly(SAMPLE_RATE / 2);
    }

    @Test
    void parse_withVad_noSegments_throwsNoSpeech() {
        StubRecognizer recognizer = new StubRecognizer();
        StubVad vad = new StubVad();

        assertThatThrownBy(() -> parse(new Inference4jAudioParser(recognizer, vad)))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("No speech detected");
        assertThat(recognizer.sliceLengths).isEmpty();
    }

    // --- parseSegments() ---

    @Test
    void parseSegments_returnsOneDocumentPerSegmentWithTimings() throws IOException {
        StubRecognizer recognizer = new StubRecognizer("first", "second");
        StubVad vad = new StubVad(
                new VoiceSegment(0.0f, 0.5f, 0.9f),
                new VoiceSegment(1.0f, 1.25f, 0.8f));

        List<Document> documents = parseSegments(new Inference4jAudioParser(recognizer, vad));

        assertThat(documents).extracting(Document::text).containsExactly("first", "second");
        Document second = documents.get(1);
        assertThat(second.metadata().getInteger(Inference4jAudioParser.SEGMENT_INDEX)).isEqualTo(1);
        assertThat(second.metadata().getFloat(Inference4jAudioParser.SEGMENT_START)).isEqualTo(1.0f);
        assertThat(second.metadata().getFloat(Inference4jAudioParser.SEGMENT_END)).isEqualTo(1.25f);
        assertThat(second.metadata().getFloat(Inference4jAudioParser.VAD_CONFIDENCE)).isEqualTo(0.8f);
    }

    @Test
    void parseSegments_skipsSegmentsWithoutSpeechButKeepsOriginalIndex() throws IOException {
        StubRecognizer recognizer = new StubRecognizer(" ", "second");
        StubVad vad = new StubVad(
                new VoiceSegment(0.0f, 0.5f, 0.9f),
                new VoiceSegment(1.0f, 1.25f, 0.8f));

        List<Document> documents = parseSegments(new Inference4jAudioParser(recognizer, vad));

        assertThat(documents).hasSize(1);
        assertThat(documents.get(0).text()).isEqualTo("second");
        assertThat(documents.get(0).metadata().getInteger(Inference4jAudioParser.SEGMENT_INDEX)).isEqualTo(1);
    }

    @Test
    void parseSegments_noSpeech_returnsEmptyList() throws IOException {
        List<Document> documents = parseSegments(
                new Inference4jAudioParser(new StubRecognizer(), new StubVad()));

        assertThat(documents).isEmpty();
    }

    @Test
    void parseSegments_withoutVad_throws() {
        Inference4jAudioParser parser = new Inference4jAudioParser(new StubRecognizer("hello"));

        assertThatThrownBy(() -> parseSegments(parser))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("VoiceActivityDetector");
    }

    @Test
    void parseSegments_deletesTempFileWhenRecognizerThrows() throws IOException {
        StubRecognizer recognizer = new StubRecognizer("first");
        recognizer.failWith = new InferenceException("model exploded");
        StubVad vad = new StubVad(new VoiceSegment(0.0f, 0.5f, 0.9f));
        List<Path> before = parserTempFiles();

        assertThatThrownBy(() -> parseSegments(new Inference4jAudioParser(recognizer, vad)))
                .isInstanceOf(InferenceException.class);
        // the VAD path never hands the recognizer a Path, so compare the temp dir instead
        assertThat(parserTempFiles()).containsExactlyInAnyOrderElementsOf(before);
    }

    // --- helpers ---

    private Document parse(Inference4jAudioParser parser) throws IOException {
        try (InputStream in = Files.newInputStream(wav)) {
            return parser.parse(in);
        }
    }

    private List<Document> parseSegments(Inference4jAudioParser parser) throws IOException {
        try (InputStream in = Files.newInputStream(wav)) {
            return parser.parseSegments(in);
        }
    }

    // --- close() ---

    @Test
    void closeClosesTheRecognizerAndTheDetector() {
        StubRecognizer recognizer = new StubRecognizer();
        StubVad vad = new StubVad();

        new Inference4jAudioParser(recognizer, vad).close();

        assertThat(recognizer.closed).isTrue();
        assertThat(vad.closed).isTrue();
    }

    @Test
    void closeWithoutVadClosesTheRecognizer() {
        StubRecognizer recognizer = new StubRecognizer();

        new Inference4jAudioParser(recognizer).close();

        assertThat(recognizer.closed).isTrue();
    }

    @Test
    void aRecognizerThatFailsToCloseStillLetsTheDetectorClose() {
        StubRecognizer recognizer = new StubRecognizer();
        recognizer.failOnClose = true;
        StubVad vad = new StubVad();

        assertThatThrownBy(() -> new Inference4jAudioParser(recognizer, vad).close())
                .isInstanceOf(InferenceException.class);
        assertThat(vad.closed).isTrue();
    }

    private static List<Path> parserTempFiles() throws IOException {
        Path tmp = Path.of(System.getProperty("java.io.tmpdir"));
        try (var files = Files.list(tmp)) {
            return files.filter(p -> p.getFileName().toString().startsWith("inference4j-audio-"))
                    .toList();
        }
    }

    /** Returns queued transcripts in order and records what it was asked to transcribe. */
    private static final class StubRecognizer implements SpeechRecognizer {

        private final Deque<String> transcripts;
        private final List<Integer> sliceLengths = new ArrayList<>();
        private int pathCalls;
        private Path lastPath;
        private long bytesSeen;
        private RuntimeException failWith;
        private boolean closed;
        private boolean failOnClose;

        StubRecognizer(String... transcripts) {
            this.transcripts = new ArrayDeque<>(List.of(transcripts));
        }

        @Override
        public Transcription transcribe(Path audioPath) {
            pathCalls++;
            lastPath = audioPath;
            try {
                bytesSeen = Files.size(audioPath);
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
            return next();
        }

        @Override
        public Transcription transcribe(float[] audioData, int sampleRate) {
            sliceLengths.add(audioData.length);
            return next();
        }

        private Transcription next() {
            if (failWith != null) {
                throw failWith;
            }
            return new Transcription(transcripts.isEmpty() ? "" : transcripts.poll());
        }

        @Override
        public void close() {
            closed = true;
            if (failOnClose) {
                throw new IllegalStateException("close failed");
            }
        }
    }

    /** Returns fixed segments; the parser feeds it decoded samples, never a path. */
    private static final class StubVad implements VoiceActivityDetector {

        private final List<VoiceSegment> segments;
        private int sampleRateSeen;
        private boolean closed;

        StubVad(VoiceSegment... segments) {
            this.segments = List.of(segments);
        }

        @Override
        public List<VoiceSegment> detect(Path audioPath) {
            return segments;
        }

        @Override
        public List<VoiceSegment> detect(float[] audioData, int sampleRate) {
            sampleRateSeen = sampleRate;
            return segments;
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
