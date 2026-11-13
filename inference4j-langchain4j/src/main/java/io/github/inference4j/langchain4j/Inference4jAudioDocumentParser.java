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
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.Metadata;
import io.github.inference4j.audio.SegmentedSpeechRecognizer;
import io.github.inference4j.audio.SpeechRecognizer;
import io.github.inference4j.audio.Transcription;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.preprocessing.audio.AudioData;
import io.github.inference4j.preprocessing.audio.AudioLoader;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A LangChain4j {@link DocumentParser} that turns audio into text {@link Document}s using
 * inference4j speech-to-text, entirely in-process.
 *
 * <p>Wrap the recognizer in a {@link SegmentedSpeechRecognizer} so long recordings are cut at pauses
 * and every piece carries its time in the recording:
 *
 * <pre>{@code
 * try (var parser = new Inference4jAudioDocumentParser(new SegmentedSpeechRecognizer(
 *         MoonshineRecognizer.builder().build(),
 *         SileroVadDetector.builder().build()))) {
 *
 *     // One document with the whole transcript
 *     Document transcript = FileSystemDocumentLoader.loadDocument(Path.of("meeting.wav"), parser);
 *
 *     // One document per segment, with start_ms / end_ms metadata for citing when something was said
 *     try (InputStream in = Files.newInputStream(Path.of("meeting.wav"))) {
 *         ingestor.ingest(parser.parseSegments(in));
 *     }
 * }
 * }</pre>
 *
 * <p>Input must be WAV (8- or 16-bit PCM, mono or stereo, any sample rate). Compressed audio and video
 * need converting first, for example with ffmpeg.
 *
 * <p>The parser owns the recognizer: {@link #close()} closes it. Thread-safe: one parser can serve a
 * whole application.
 */
public class Inference4jAudioDocumentParser implements DocumentParser, AutoCloseable {

    /** Position of the segment in the recording, from 0 ({@code int}). */
    public static final String SEGMENT_INDEX = "segment_index";

    /** Where the segment starts in the recording, in milliseconds ({@code long}). */
    public static final String START_MS = "start_ms";

    /** Where the segment ends in the recording, in milliseconds ({@code long}). */
    public static final String END_MS = "end_ms";

    private final SpeechRecognizer recognizer;

    /**
     * @param recognizer the recognizer to transcribe with, typically a {@link SegmentedSpeechRecognizer};
     *                   closed by {@link #close()}
     */
    public Inference4jAudioDocumentParser(SpeechRecognizer recognizer) {
        this.recognizer = Objects.requireNonNull(recognizer, "recognizer is required");
    }

    /**
     * Transcribes the audio into one {@link Document} holding the whole transcript.
     *
     * @throws InferenceException if no speech is found (LangChain4j documents cannot be blank), or the
     *                            stream is not readable WAV
     */
    @Override
    public Document parse(InputStream inputStream) {
        Transcription transcription = transcribe(inputStream).transcription();
        String text = transcription.text();
        if (text == null || text.isBlank()) {
            throw new InferenceException("No speech detected in audio");
        }
        return Document.from(text.strip());
    }

    /**
     * Transcribes the audio into one {@link Document} per timed segment, in recording order, each with
     * {@value #SEGMENT_INDEX}, {@value #START_MS} and {@value #END_MS} metadata. Not part of LangChain4j's
     * {@link DocumentParser} contract: call it directly and pass the result to
     * {@code EmbeddingStoreIngestor.ingest(List<Document>)}.
     *
     * <p>Segments come from the recognizer; {@link SegmentedSpeechRecognizer} gives one per piece of
     * audio. A recognizer that returns none yields a single document spanning the whole recording.
     * Segments without speech are skipped, so silent audio gives an empty list.
     *
     * @throws InferenceException if the stream is not readable WAV
     */
    public List<Document> parseSegments(InputStream inputStream) {
        TranscribedAudio audio = transcribe(inputStream);
        List<Transcription.Segment> segments = audio.transcription().segments();
        if (segments.isEmpty()) {
            segments = List.of(new Transcription.Segment(audio.transcription().text(), 0f, audio.seconds()));
        }

        List<Document> documents = new ArrayList<>(segments.size());
        for (Transcription.Segment segment : segments) {
            if (segment.text() == null || segment.text().isBlank()) {
                continue;
            }
            Metadata metadata = new Metadata()
                    .put(SEGMENT_INDEX, documents.size())
                    .put(START_MS, Math.round(segment.startTime() * 1000d))
                    .put(END_MS, Math.round(segment.endTime() * 1000d));
            documents.add(Document.from(segment.text().strip(), metadata));
        }
        return documents;
    }

    /**
     * Closes the recognizer.
     */
    @Override
    public void close() {
        recognizer.close();
    }

    private TranscribedAudio transcribe(InputStream inputStream) {
        Path path;
        try {
            path = Files.createTempFile("inference4j-audio-", ".wav");
        } catch (IOException e) {
            throw new InferenceException("Failed to create temp file for audio: " + e.getMessage(), e);
        }
        try {
            Files.copy(inputStream, path, StandardCopyOption.REPLACE_EXISTING);
            AudioData audio = AudioLoader.load(path);
            Transcription transcription = recognizer.transcribe(audio.samples(), audio.sampleRate());
            return new TranscribedAudio(transcription, (float) audio.samples().length / audio.sampleRate());
        } catch (IOException e) {
            throw new InferenceException("Failed to read audio stream: " + e.getMessage(), e);
        } finally {
            try {
                Files.deleteIfExists(path);
            } catch (IOException ignored) {
                // best effort; the file is in the temp directory
            }
        }
    }

    private record TranscribedAudio(Transcription transcription, float seconds) {
    }
}
