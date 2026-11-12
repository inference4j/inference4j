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
import io.github.inference4j.audio.SpeechRecognizer;
import io.github.inference4j.audio.VoiceActivityDetector;
import io.github.inference4j.audio.VoiceSegment;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.preprocessing.audio.AudioData;
import io.github.inference4j.preprocessing.audio.AudioLoader;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * A LangChain4j {@link DocumentParser} that turns audio into a text {@link Document}
 * using inference4j speech-to-text, entirely in-process.
 *
 * <pre>{@code
 * try (var parser = new Inference4jAudioParser(
 *         Wav2Vec2Recognizer.builder().build(),
 *         SileroVadDetector.builder().build())) {
 *     Document transcript = FileSystemDocumentLoader.loadDocument(Path.of("meeting.wav"), parser);
 * }
 * }</pre>
 *
 * <p>The parser owns the recognizer and detector: {@link #close()} closes them.
 *
 * <p>Input must be WAV (8- or 16-bit PCM, mono or stereo, any sample rate).
 *
 * <p>Thread-safe: one parser can serve a whole application.
 */
public class Inference4jAudioParser implements DocumentParser, AutoCloseable {

    private final SpeechRecognizer recognizer;
    private final VoiceActivityDetector vad;      // null when VAD is disabled

    /**
     * Transcribes the whole audio, without voice activity detection.
     */
    public Inference4jAudioParser(SpeechRecognizer recognizer) {
        this(recognizer, null);
    }

    /**
     * Transcribes only the voiced segments found by {@code vad}.
     */
    public Inference4jAudioParser(SpeechRecognizer recognizer, VoiceActivityDetector vad) {
        this.recognizer = recognizer;
        this.vad = vad;
    }

    /**
     * Transcribes the audio into a single {@link dev.langchain4j.data.document.Document}. With VAD, the voiced segments
     * are transcribed one by one and joined with spaces, and the document carries a
     * {@value #VAD_SEGMENT_COUNT} metadata entry.
     *
     * @throws InferenceException if no speech is found (LangChain4j documents cannot be blank)
     */
    @Override
    public Document parse(InputStream inputStream) {
        return withTempFile(inputStream, path -> {
            if (vad == null) {
                return Document.from(requireSpeech(recognizer.transcribe(path).text()));
            }
            List<SegmentTranscript> transcripts = transcribeVoicedSegments(path);
            String text = transcripts.stream()
                    .map(SegmentTranscript::text)
                    .collect(Collectors.joining(" "));
            return Document.from(requireSpeech(text),
                    Metadata.from(VAD_SEGMENT_COUNT, String.valueOf(transcripts.size())));
        });
    }

    /**
     * Transcribes each voiced segment into its own {@link Document}, carrying
     * {@value #SEGMENT_INDEX}, {@value #SEGMENT_START}, {@value #SEGMENT_END} and
     * {@value #VAD_CONFIDENCE} metadata. Not part of LangChain4j's {@link DocumentParser}
     * contract — call it directly and pass the result to
     * {@code EmbeddingStoreIngestor.ingest(List<Document>)}.
     *
     * <p>Segments with no recognizable speech are skipped, so the list can be empty.
     *
     * @throws IllegalStateException if this parser was created without a voice activity detector
     */
    public List<Document> parseSegments(InputStream inputStream) {
        if (vad == null) {
            throw new IllegalStateException("parseSegments requires a VoiceActivityDetector — " + "use new Inference4jAudioParser(recognizer, vad)");
        }
        return withTempFile(inputStream, path -> transcribeVoicedSegments(path).stream()
                .map(SegmentTranscript::toDocument)
                .toList());
    }

    /**
     * Closes the recognizer and the voice activity detector.
     *
     * @throws InferenceException if either fails to close; the other is still closed
     */
    @Override
    public void close() {
        Exception failure = closeQuietly(recognizer, null);
        failure = closeQuietly(vad, failure);
        if (failure != null) {
            throw new InferenceException("Failed to close the audio parser: " + failure.getMessage(), failure);
        }
    }

    private static Exception closeQuietly(AutoCloseable resource, Exception failure) {
        if (resource == null) {
            return failure;
        }
        try {
            resource.close();
        } catch (Exception e) {
            if (failure == null) {
                return e;
            }
            failure.addSuppressed(e);
        }
        return failure;
    }

    public static final String VAD_SEGMENT_COUNT = "vad_segment_count";
    public static final String SEGMENT_INDEX = "segment_index";
    public static final String SEGMENT_START = "segment_start_seconds";
    public static final String SEGMENT_END = "segment_end_seconds";
    public static final String VAD_CONFIDENCE = "vad_confidence";

    private List<SegmentTranscript> transcribeVoicedSegments(Path path) {
        AudioData audio = AudioLoader.load(path);
        List<VoiceSegment> segments = vad.detect(audio.samples(), audio.sampleRate());

        List<SegmentTranscript> transcripts = new ArrayList<>();
        for (int i = 0; i < segments.size(); i++) {
            VoiceSegment segment = segments.get(i);
            float[] slice = slice(audio, segment);
            if (slice.length == 0) {
                continue;
            }
            String text = recognizer.transcribe(slice, audio.sampleRate()).text().strip();
            if (!text.isEmpty()) {
                transcripts.add(new SegmentTranscript(i, segment, text));
            }
        }
        return transcripts;
    }

    private static float[] slice(AudioData audio, VoiceSegment segment) {
        int length = audio.samples().length;
        int from = Math.min(length, Math.max(0, Math.round(segment.start() * audio.sampleRate())));
        int to = Math.min(length, Math.max(from, Math.round(segment.end() * audio.sampleRate())));
        return Arrays.copyOfRange(audio.samples(), from, to);
    }

    private static String requireSpeech(String text) {
        if (text == null || text.isBlank()) {
            throw new InferenceException("No speech detected in audio");
        }
        return text.strip();
    }

    private static <T> T withTempFile(InputStream inputStream, Function<Path, T> action) {
        Path path;
        try {
            path = Files.createTempFile("inference4j-audio-", ".wav");
        } catch (IOException e) {
            throw new InferenceException("Failed to create temp file for audio: " + e.getMessage(), e);
        }
        try {
            Files.copy(inputStream, path, StandardCopyOption.REPLACE_EXISTING);
            return action.apply(path);
        } catch (IOException e) {
            throw new InferenceException("Failed to read audio stream: " + e.getMessage(), e);
        } finally {
            try {
                Files.deleteIfExists(path);
            } catch (IOException ignored) {
            }
        }
    }

    private record SegmentTranscript(int index, VoiceSegment segment, String text) {

        Document toDocument() {
            Metadata metadata = new Metadata()
                    .put(SEGMENT_INDEX, index)
                    .put(SEGMENT_START, segment.start())
                    .put(SEGMENT_END, segment.end())
                    .put(VAD_CONFIDENCE, segment.confidence());
            return Document.from(text, metadata);
        }
    }
}
