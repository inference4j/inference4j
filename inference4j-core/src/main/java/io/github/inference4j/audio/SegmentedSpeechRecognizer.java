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

import io.github.inference4j.preprocessing.audio.AudioData;
import io.github.inference4j.preprocessing.audio.AudioLoader;

import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * A {@link SpeechRecognizer} that splits long audio at pauses in speech before transcribing it.
 *
 * <p>Some recognizers only see a fixed window of audio at a time; Whisper, for example, processes
 * 30 seconds per pass. Cutting long recordings at fixed intervals splits words and sentences, which
 * makes such models drop or repeat text around the cuts. This recognizer asks a
 * {@link VoiceActivityDetector} where the pauses are and cuts there instead:
 *
 * <pre>{@code
 * try (SpeechRecognizer recognizer = new SegmentedSpeechRecognizer(
 *         WhisperSpeechModel.builder().modelId("inference4j/whisper-small-genai").build(),
 *         SileroVadDetector.builder().build())) {
 *     String text = recognizer.transcribe(Path.of("meeting.wav")).text();
 * }
 * }</pre>
 *
 * <p>Each piece ends at the midpoint of the latest pause that fits within the maximum segment
 * length (30 seconds by default). When a stretch of speech has no pause, it is cut at the maximum
 * length. Audio that already fits is passed to the wrapped recognizer unchanged. The pieces'
 * transcripts are joined with spaces, and their segments are shifted to the recording's timeline.
 *
 * <p>Closing this recognizer closes both the wrapped recognizer and the voice activity detector.
 */
public class SegmentedSpeechRecognizer implements SpeechRecognizer {

    private static final Duration DEFAULT_MAX_SEGMENT = Duration.ofSeconds(30);
    private static final Duration MIN_SEGMENT = Duration.ofSeconds(1);

    private final SpeechRecognizer recognizer;
    private final VoiceActivityDetector vad;
    private final Duration maxSegment;

    /**
     * Creates a recognizer that cuts audio into pieces of at most 30 seconds.
     */
    public SegmentedSpeechRecognizer(SpeechRecognizer recognizer, VoiceActivityDetector vad) {
        this(recognizer, vad, DEFAULT_MAX_SEGMENT);
    }

    /**
     * Creates a recognizer that cuts audio into pieces of at most {@code maxSegment}.
     *
     * @throws IllegalArgumentException if {@code maxSegment} is not longer than one second
     */
    public SegmentedSpeechRecognizer(SpeechRecognizer recognizer, VoiceActivityDetector vad, Duration maxSegment) {
        this.recognizer = Objects.requireNonNull(recognizer, "recognizer");
        this.vad = Objects.requireNonNull(vad, "vad");
        this.maxSegment = Objects.requireNonNull(maxSegment, "maxSegment");
        if (maxSegment.compareTo(MIN_SEGMENT) <= 0) {
            throw new IllegalArgumentException("maxSegment must be longer than " + MIN_SEGMENT + ", got " + maxSegment);
        }
    }

    @Override
    public Transcription transcribe(Path audioPath) {
        AudioData audio = AudioLoader.load(audioPath);
        return transcribe(audio.samples(), audio.sampleRate());
    }

    @Override
    public Transcription transcribe(float[] audioData, int sampleRate) {
        int maxSamples = toSamples(maxSegment, sampleRate);
        if (audioData.length <= maxSamples) {
            return recognizer.transcribe(audioData, sampleRate);
        }

        List<Integer> pauses = pauseMidpoints(vad.detect(audioData, sampleRate), sampleRate);
        int minSamples = toSamples(MIN_SEGMENT, sampleRate);
        StringBuilder text = new StringBuilder();
        List<Transcription.Segment> segments = new ArrayList<>();

        int start = 0;
        while (start < audioData.length) {
            int end = audioData.length - start <= maxSamples
                    ? audioData.length
                    : latestPause(pauses, start + minSamples, start + maxSamples);
            Transcription piece = recognizer.transcribe(Arrays.copyOfRange(audioData, start, end), sampleRate);
            if (piece.text() != null && !piece.text().isBlank()) {
                if (!text.isEmpty()) {
                    text.append(' ');
                }
                text.append(piece.text().strip());
            }
            float offset = (float) start / sampleRate;
            for (Transcription.Segment s : piece.segments()) {
                segments.add(new Transcription.Segment(s.text(), s.startTime() + offset, s.endTime() + offset));
            }
            start = end;
        }
        return new Transcription(text.toString(), segments);
    }

    /** Sample positions halfway between consecutive speech segments. */
    private static List<Integer> pauseMidpoints(List<VoiceSegment> speech, int sampleRate) {
        List<Integer> pauses = new ArrayList<>();
        for (int i = 0; i + 1 < speech.size(); i++) {
            pauses.add(Math.round((speech.get(i).end() + speech.get(i + 1).start()) / 2 * sampleRate));
        }
        return pauses;
    }

    /** The latest pause in {@code (from, to]}, or {@code to} when there is none. */
    private static int latestPause(List<Integer> pauses, int from, int to) {
        int cut = to;
        for (int pause : pauses) {
            if (pause > from && pause <= to) {
                cut = pause;
            }
        }
        return cut;
    }

    private static int toSamples(Duration duration, int sampleRate) {
        return (int) (duration.toMillis() * sampleRate / 1000);
    }

    @Override
    public void close() {
        try {
            recognizer.close();
        } finally {
            vad.close();
        }
    }
}
