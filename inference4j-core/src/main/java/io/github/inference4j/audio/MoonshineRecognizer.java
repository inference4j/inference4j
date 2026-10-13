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
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.audio.AudioData;
import io.github.inference4j.preprocessing.audio.AudioLoader;
import io.github.inference4j.preprocessing.audio.AudioTransformPipeline;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.tokenizer.SentencePieceBpeTokenizer;
import io.github.inference4j.tokenizer.TokenDecoder;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Speech-to-text with Moonshine v2, an English ASR model built for live and on-device use.
 *
 * <p>Unlike Whisper, Moonshine spends compute only on the audio it is given: there is no padded 30-second
 * window, so a 3-second clip costs about 3 seconds' worth of work. Moonshine AI reports, on the Open ASR
 * Leaderboard average, 7.84% word error rate for the small model and 6.65% for the medium model, against
 * 7.44% for Whisper Large v3.
 *
 * <pre>{@code
 * try (MoonshineRecognizer recognizer = MoonshineRecognizer.builder().build()) {
 *     Transcription result = recognizer.transcribe(Path.of("meeting.wav"));
 *     System.out.println(result.text());
 * }
 * }</pre>
 *
 * <p>One call transcribes at most {@value #MAX_SECONDS} seconds: beyond that the model starts repeating
 * itself. For longer recordings, cut the audio at pauses with {@link SegmentedSpeechRecognizer}:
 *
 * <pre>{@code
 * SpeechRecognizer recognizer = new SegmentedSpeechRecognizer(
 *         MoonshineRecognizer.builder().build(),
 *         SileroVadDetector.builder().build(),
 *         Duration.ofSeconds(15));
 * }</pre>
 *
 * <h2>Models</h2>
 * <table>
 *   <caption>Moonshine v2 models</caption>
 *   <tr><th>Model ID</th><th>Parameters</th><th>Download</th><th>WER (Open ASR avg., Moonshine AI)</th></tr>
 *   <tr><td>{@code inference4j/moonshine-streaming-tiny-en}</td><td>34M</td><td>46 MB</td><td>12.00%</td></tr>
 *   <tr><td>{@code inference4j/moonshine-streaming-small-en} (default)</td><td>123M</td><td>140 MB</td><td>7.84%</td></tr>
 *   <tr><td>{@code inference4j/moonshine-streaming-medium-en}</td><td>245M</td><td>260 MB</td><td>6.65%</td></tr>
 * </table>
 *
 * <p>The models are Moonshine AI's 8-bit quantized ONNX Runtime ({@code .ort}) graphs, MIT licensed.
 *
 * @see SegmentedSpeechRecognizer
 */
public class MoonshineRecognizer implements SpeechRecognizer {

    /** The longest audio one call accepts, in seconds. */
    public static final int MAX_SECONDS = 30;

    private static final String DEFAULT_MODEL_ID = "inference4j/moonshine-streaming-small-en";

    /** Token budget per second of audio; stops a decode that has started looping. */
    private static final double MAX_TOKENS_PER_SECOND = 6.5;

    private static final int MAX_TOKENS = 256;

    private final MoonshineModel model;
    private final TokenDecoder decoder;
    private final AudioTransformPipeline pipeline;

    /** Serializes all use of the model's sessions, by this recognizer and its streams. */
    private final Object lock = new Object();

    private MoonshineRecognizer(MoonshineModel model, TokenDecoder decoder) {
        this.model = model;
        this.decoder = decoder;
        this.pipeline = AudioTransformPipeline.builder()
                .resample(MoonshineModel.SAMPLE_RATE)
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public Transcription transcribe(Path audioPath) {
        AudioData audio = AudioLoader.load(audioPath);
        return transcribe(audio.samples(), audio.sampleRate());
    }

    /**
     * {@inheritDoc}
     *
     * @throws IllegalArgumentException if the audio is longer than {@value #MAX_SECONDS} seconds
     */
    @Override
    public Transcription transcribe(float[] audioData, int sampleRate) {
        Duration length = Duration.ofMillis(Math.round(audioData.length * 1000.0 / sampleRate));
        if (length.compareTo(Duration.ofSeconds(MAX_SECONDS)) > 0) {
            throw new IllegalArgumentException("Moonshine transcribes at most " + MAX_SECONDS
                    + " seconds per call, got " + length.toMillis() / 1000.0
                    + " s; wrap the recognizer in a SegmentedSpeechRecognizer to cut long audio at pauses");
        }
        float[] samples = pipeline.transform(new AudioData(audioData, sampleRate)).samples();
        synchronized (lock) {
            try (MoonshineModel.Line line = model.newLine()) {
                line.addAudio(samples);
                line.encode(true);
                int[] tokens = line.decode(maxTokens(samples.length), new int[0]);
                return new Transcription(decoder.decode(tokens).strip());
            }
        }
    }

    /**
     * Starts building a live transcription stream, which splits continuous audio into lines at the pauses
     * {@code vad} finds and transcribes each line while it is being spoken.
     *
     * <pre>{@code
     * try (TranscriptionStream stream = recognizer.stream(vad)
     *         .listener(new TranscriptListener() {
     *             public void onLineCompleted(TranscriptLine line) {
     *                 System.out.println(line.text());
     *             }
     *         })
     *         .build()) {
     *     stream.addAudio(samples, 48000);
     * }
     * }</pre>
     *
     * <p>The stream uses this recognizer and {@code vad} but does not own them: closing the stream leaves
     * both open, and one recognizer can serve several streams. Work for streams of the same recognizer is
     * serialized.
     *
     * @param vad finds the pauses that end lines; each stream keeps its own detection state
     */
    public StreamBuilder stream(SileroVadDetector vad) {
        return new StreamBuilder(Objects.requireNonNull(vad, "vad"));
    }

    /** The token budget for audio of this many samples at 16 kHz. */
    static int maxTokens(int samples) {
        double seconds = (double) samples / MoonshineModel.SAMPLE_RATE;
        return Math.min((int) Math.ceil(seconds * MAX_TOKENS_PER_SECOND), MAX_TOKENS);
    }

    @Override
    public void close() {
        model.close();
    }

    /** Configures a {@link TranscriptionStream}. */
    public class StreamBuilder {

        private final SileroVadDetector vad;
        private Duration updateInterval = Duration.ofMillis(500);
        private Duration maxLineDuration = Duration.ofSeconds(15);
        private TranscriptListener listener = new TranscriptListener() {
        };

        private StreamBuilder(SileroVadDetector vad) {
            this.vad = vad;
        }

        /**
         * How much audio passes between updates of the open line's text; defaults to 500 ms. Shorter
         * intervals give livelier text at the cost of more decoding.
         */
        public StreamBuilder updateInterval(Duration updateInterval) {
            if (updateInterval.isNegative() || updateInterval.isZero()) {
                throw new IllegalArgumentException("updateInterval must be positive, got " + updateInterval);
            }
            this.updateInterval = updateInterval;
            return this;
        }

        /**
         * The longest a line can get when the speaker does not pause; defaults to 15 seconds, at most
         * {@value MoonshineRecognizer#MAX_SECONDS}. Past two thirds of it, lines end at the next dip in speech.
         */
        public StreamBuilder maxLineDuration(Duration maxLineDuration) {
            if (maxLineDuration.compareTo(Duration.ofSeconds(1)) < 0
                    || maxLineDuration.compareTo(Duration.ofSeconds(MAX_SECONDS)) > 0) {
                throw new IllegalArgumentException("maxLineDuration must be between 1 and " + MAX_SECONDS
                        + " seconds, got " + maxLineDuration);
            }
            this.maxLineDuration = maxLineDuration;
            return this;
        }

        /** Receives the stream's events. */
        public StreamBuilder listener(TranscriptListener listener) {
            this.listener = Objects.requireNonNull(listener, "listener");
            return this;
        }

        public TranscriptionStream build() {
            return new MoonshineTranscriptionStream(model, decoder, lock, vad, updateInterval, maxLineDuration,
                    listener);
        }
    }

    public static class Builder {

        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;

        /** Where to find the model; defaults to HuggingFace. */
        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        /** The model to load; defaults to {@code inference4j/moonshine-streaming-small-en}. */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /** Customizes the ONNX Runtime options of every graph. */
        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public MoonshineRecognizer build() {
            ModelSource source = modelSource != null ? modelSource : HuggingFaceModelSource.defaultInstance();
            String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
            List<String> files = new ArrayList<>(MoonshineModel.FILES);
            files.add("tokenizer.json");
            Path dir = source.resolve(id, files);
            for (String file : files) {
                if (!Files.exists(dir.resolve(file))) {
                    throw new ModelSourceException("Model file not found: " + dir.resolve(file));
                }
            }
            TokenDecoder decoder = SentencePieceBpeTokenizer.provider().create(dir, List.of()).decoder();
            MoonshineModel model = MoonshineModel.load(dir, path -> sessionConfigurer != null
                    ? InferenceSession.create(path, sessionConfigurer)
                    : InferenceSession.create(path));
            return new MoonshineRecognizer(model, decoder);
        }
    }
}
