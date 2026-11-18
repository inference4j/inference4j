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
 * <p>Pieces of a few seconds often hold too little text to match a question. {@link Builder#mergeSegments(int)}
 * joins consecutive segments into excerpts of up to a number of characters, each still pointing to where it starts:
 *
 * <pre>{@code
 * var parser = Inference4jAudioDocumentParser.builder()
 *         .recognizer(new SegmentedSpeechRecognizer(
 *                 MoonshineRecognizer.builder().build(),
 *                 SileroVadDetector.builder().build(),
 *                 Duration.ofSeconds(8)))
 *         .mergeSegments(300)
 *         .build();
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
    private final int mergeMaxChars;          // 0: one document per segment

    /**
     * @param recognizer the recognizer to transcribe with, typically a {@link SegmentedSpeechRecognizer};
     *                   closed by {@link #close()}
     */
    public Inference4jAudioDocumentParser(SpeechRecognizer recognizer) {
        this(recognizer, 0);
    }

    private Inference4jAudioDocumentParser(SpeechRecognizer recognizer, int mergeMaxChars) {
        this.recognizer = Objects.requireNonNull(recognizer, "recognizer is required");
        this.mergeMaxChars = mergeMaxChars;
    }

    public static Builder builder() {
        return new Builder();
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
     * <p>With {@link Builder#mergeSegments(int)}, consecutive segments are joined with a space until the next one
     * would make the text longer than the limit. A joined document has the {@value #START_MS} of its first segment
     * and the {@value #END_MS} of its last; {@value #SEGMENT_INDEX} counts the joined documents. A single segment
     * longer than the limit stays whole.
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
        StringBuilder text = new StringBuilder();
        float start = 0;
        float end = 0;
        for (Transcription.Segment segment : segments) {
            if (segment.text() == null || segment.text().isBlank()) {
                continue;
            }
            String piece = segment.text().strip();
            boolean fits = mergeMaxChars > 0 && text.length() + 1 + piece.length() <= mergeMaxChars;
            if (!text.isEmpty() && !fits) {
                documents.add(document(text, documents.size(), start, end));
                text.setLength(0);
            }
            if (text.isEmpty()) {
                start = segment.startTime();
            } else {
                text.append(' ');
            }
            text.append(piece);
            end = segment.endTime();
        }
        if (!text.isEmpty()) {
            documents.add(document(text, documents.size(), start, end));
        }
        return documents;
    }

    private static Document document(CharSequence text, int index, float startSeconds, float endSeconds) {
        Metadata metadata = new Metadata()
                .put(SEGMENT_INDEX, index)
                .put(START_MS, Math.round(startSeconds * 1000d))
                .put(END_MS, Math.round(endSeconds * 1000d));
        return Document.from(text.toString(), metadata);
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

    /**
     * Builder for {@link Inference4jAudioDocumentParser}.
     */
    public static class Builder {

        private SpeechRecognizer recognizer;
        private int mergeMaxChars;

        private Builder() {
        }

        /**
         * The recognizer to transcribe with, typically a {@link SegmentedSpeechRecognizer}. Required; closed by
         * the parser's {@link Inference4jAudioDocumentParser#close()}.
         */
        public Builder recognizer(SpeechRecognizer recognizer) {
            this.recognizer = recognizer;
            return this;
        }

        /**
         * Joins consecutive segments returned by
         * {@link Inference4jAudioDocumentParser#parseSegments(InputStream)} into documents of up to
         * {@code maxChars} characters. Off by default: one document per segment.
         */
        public Builder mergeSegments(int maxChars) {
            if (maxChars < 1) {
                throw new IllegalArgumentException("maxChars must be at least 1");
            }
            this.mergeMaxChars = maxChars;
            return this;
        }

        /**
         * @throws NullPointerException if no recognizer is set
         */
        public Inference4jAudioDocumentParser build() {
            return new Inference4jAudioDocumentParser(recognizer, mergeMaxChars);
        }
    }
}
