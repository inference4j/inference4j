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
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import io.github.inference4j.audio.SegmentedSpeechRecognizer;
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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static io.github.inference4j.langchain4j.Inference4jAudioDocumentParser.END_MS;
import static io.github.inference4j.langchain4j.Inference4jAudioDocumentParser.SEGMENT_INDEX;
import static io.github.inference4j.langchain4j.Inference4jAudioDocumentParser.START_MS;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class Inference4jAudioDocumentParserTest {

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
        wav = tempDir.resolve("meeting.wav");
        AudioWriter.write(new AudioData(samples, SAMPLE_RATE), wav);
    }

    // --- parse() ---

    @Test
    void parseReturnsTheWholeTranscriptAsOneDocument() throws IOException {
        var recognizer = new StubRecognizer(new Transcription("  hello world  "));

        Document document = parse(new Inference4jAudioDocumentParser(recognizer));

        assertThat(document.text()).isEqualTo("hello world");
    }

    @Test
    void theRecognizerGetsTheDecodedSamples() throws IOException {
        var recognizer = new StubRecognizer(new Transcription("hello"));

        parse(new Inference4jAudioDocumentParser(recognizer));

        assertThat(recognizer.samplesSeen).isEqualTo(SAMPLE_RATE * 2);
        assertThat(recognizer.sampleRateSeen).isEqualTo(SAMPLE_RATE);
    }

    @Test
    void parseRejectsAudioWithoutSpeech() {
        var recognizer = new StubRecognizer(new Transcription("   "));

        assertThatThrownBy(() -> parse(new Inference4jAudioDocumentParser(recognizer)))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("No speech detected");
    }

    @Test
    void worksWithFileSystemDocumentLoader() {
        var parser = new Inference4jAudioDocumentParser(new StubRecognizer(new Transcription("hello")));

        Document document = FileSystemDocumentLoader.loadDocument(wav, parser);

        assertThat(document.text()).isEqualTo("hello");
        assertThat(document.metadata().getString(Document.FILE_NAME)).isEqualTo("meeting.wav");
    }

    // --- parseSegments() ---

    @Test
    void parseSegmentsGivesOneDocumentPerSegmentWithMillisecondTimes() throws IOException {
        var recognizer = new StubRecognizer(new Transcription("Marie Curie moved to Paris. She won.", List.of(
                new Transcription.Segment(" Marie Curie moved to Paris. ", 0f, 1.25f),
                new Transcription.Segment("She won.", 1.25f, 2f))));

        List<Document> documents = parseSegments(new Inference4jAudioDocumentParser(recognizer));

        assertThat(documents).extracting(Document::text)
                .containsExactly("Marie Curie moved to Paris.", "She won.");
        assertThat(documents.get(0).metadata().getInteger(SEGMENT_INDEX)).isZero();
        assertThat(documents.get(0).metadata().getLong(START_MS)).isZero();
        assertThat(documents.get(0).metadata().getLong(END_MS)).isEqualTo(1250L);
        assertThat(documents.get(1).metadata().getInteger(SEGMENT_INDEX)).isEqualTo(1);
        assertThat(documents.get(1).metadata().getLong(START_MS)).isEqualTo(1250L);
        assertThat(documents.get(1).metadata().getLong(END_MS)).isEqualTo(2000L);
    }

    @Test
    void parseSegmentsSkipsBlankSegmentsAndKeepsIndexesContiguous() throws IOException {
        var recognizer = new StubRecognizer(new Transcription("first second", List.of(
                new Transcription.Segment("first", 0f, 0.5f),
                new Transcription.Segment("  ", 0.5f, 1f),
                new Transcription.Segment("second", 1f, 2f))));

        List<Document> documents = parseSegments(new Inference4jAudioDocumentParser(recognizer));

        assertThat(documents).extracting(Document::text).containsExactly("first", "second");
        assertThat(documents.get(1).metadata().getInteger(SEGMENT_INDEX)).isEqualTo(1);
        assertThat(documents.get(1).metadata().getLong(START_MS)).isEqualTo(1000L);
    }

    @Test
    void withoutRecognizerSegmentsOneDocumentSpansTheRecording() throws IOException {
        var recognizer = new StubRecognizer(new Transcription(" hello "));

        List<Document> documents = parseSegments(new Inference4jAudioDocumentParser(recognizer));

        assertThat(documents).singleElement().satisfies(document -> {
            assertThat(document.text()).isEqualTo("hello");
            assertThat(document.metadata().getLong(START_MS)).isZero();
            assertThat(document.metadata().getLong(END_MS)).isEqualTo(2000L);
        });
    }

    @Test
    void parseSegmentsOfSilenceIsEmpty() throws IOException {
        var recognizer = new StubRecognizer(new Transcription(""));

        assertThat(parseSegments(new Inference4jAudioDocumentParser(recognizer))).isEmpty();
    }

    @Test
    void segmentedSpeechRecognizerSuppliesTheSegments() throws IOException {
        var segmented = new SegmentedSpeechRecognizer(
                new StubRecognizer(new Transcription("hello")), new StubVad());

        List<Document> documents = parseSegments(new Inference4jAudioDocumentParser(segmented));

        assertThat(documents).singleElement().satisfies(document -> {
            assertThat(document.text()).isEqualTo("hello");
            assertThat(document.metadata().getLong(END_MS)).isEqualTo(2000L);
        });
    }

    // --- input handling ---

    @Test
    void theTempFileIsDeletedAfterParsing() throws IOException {
        List<Path> before = parserTempFiles();

        parse(new Inference4jAudioDocumentParser(new StubRecognizer(new Transcription("hello"))));

        assertThat(parserTempFiles()).containsExactlyInAnyOrderElementsOf(before);
    }

    @Test
    void theTempFileIsDeletedWhenTheRecognizerFails() throws IOException {
        var recognizer = new StubRecognizer(new Transcription("hello"));
        recognizer.failWith = new InferenceException("model exploded");
        List<Path> before = parserTempFiles();

        assertThatThrownBy(() -> parseSegments(new Inference4jAudioDocumentParser(recognizer)))
                .isInstanceOf(InferenceException.class)
                .hasMessage("model exploded");
        assertThat(parserTempFiles()).containsExactlyInAnyOrderElementsOf(before);
    }

    @Test
    void nonWavInputFails() {
        var parser = new Inference4jAudioDocumentParser(new StubRecognizer(new Transcription("hello")));
        InputStream notWav = new ByteArrayInputStream("not audio".getBytes(StandardCharsets.UTF_8));

        assertThatThrownBy(() -> parser.parse(notWav))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("Unsupported audio format");
    }

    // --- lifecycle ---

    @Test
    void aRecognizerIsRequired() {
        assertThatThrownBy(() -> new Inference4jAudioDocumentParser(null))
                .isInstanceOf(NullPointerException.class)
                .hasMessageContaining("recognizer");
    }

    @Test
    void closeClosesTheRecognizer() {
        var recognizer = new StubRecognizer(new Transcription("hello"));

        new Inference4jAudioDocumentParser(recognizer).close();

        assertThat(recognizer.closed).isTrue();
    }

    // --- helpers ---

    private Document parse(Inference4jAudioDocumentParser parser) throws IOException {
        try (InputStream in = Files.newInputStream(wav)) {
            return parser.parse(in);
        }
    }

    private List<Document> parseSegments(Inference4jAudioDocumentParser parser) throws IOException {
        try (InputStream in = Files.newInputStream(wav)) {
            return parser.parseSegments(in);
        }
    }

    private static List<Path> parserTempFiles() throws IOException {
        Path tmp = Path.of(System.getProperty("java.io.tmpdir"));
        try (var files = Files.list(tmp)) {
            return files.filter(p -> p.getFileName().toString().startsWith("inference4j-audio-")).toList();
        }
    }

    /** Returns a fixed transcription and records what it was given. */
    private static final class StubRecognizer implements SpeechRecognizer {

        private final Transcription result;
        private int samplesSeen;
        private int sampleRateSeen;
        private RuntimeException failWith;
        private boolean closed;

        StubRecognizer(Transcription result) {
            this.result = result;
        }

        @Override
        public Transcription transcribe(Path audioPath) {
            throw new UnsupportedOperationException("the parser passes decoded samples");
        }

        @Override
        public Transcription transcribe(float[] audioData, int sampleRate) {
            if (failWith != null) {
                throw failWith;
            }
            samplesSeen = audioData.length;
            sampleRateSeen = sampleRate;
            return result;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    /** Finds no speech; audio short enough for one piece never reaches it anyway. */
    private static final class StubVad implements VoiceActivityDetector {

        @Override
        public List<VoiceSegment> detect(Path audioPath) {
            return List.of();
        }

        @Override
        public List<VoiceSegment> detect(float[] audioData, int sampleRate) {
            return List.of();
        }

        @Override
        public void close() {
        }
    }
}
