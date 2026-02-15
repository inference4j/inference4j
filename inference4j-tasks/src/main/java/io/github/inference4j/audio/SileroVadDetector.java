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

import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Silero Voice Activity Detection (VAD) detector.
 *
 * <p>Silero VAD is a lightweight, accurate voice activity detector that identifies
 * speech segments in audio. It runs in real-time and is well-suited for
 * preprocessing audio before transcription or detecting speech boundaries.
 *
 * <p><strong>Note:</strong> This class does not extend {@link io.github.inference4j.AbstractInferenceTask}
 * because it is a stateful model that makes multiple {@code session.run()} calls per input
 * (sliding window with hidden state carried across frames).
 *
 * <h2>Target model</h2>
 * <p>Designed for <a href="https://github.com/snakers4/silero-vad">Silero VAD</a>
 * ONNX models. The model expects 16kHz mono audio and outputs per-frame speech
 * probabilities.
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (SileroVadDetector vad = SileroVadDetector.builder().build()) {
 *     List<VoiceSegment> segments = vad.detect(Path.of("audio.wav"));
 *     for (VoiceSegment segment : segments) {
 *         System.out.printf("Speech: %.2fs - %.2fs (confidence: %.2f)%n",
 *             segment.start(), segment.end(), segment.confidence());
 *     }
 * }
 * }</pre>
 *
 * <h2>Custom configuration</h2>
 * <pre>{@code
 * try (SileroVadDetector vad = SileroVadDetector.builder()
 *         .modelId("my-org/my-silero-vad")
 *         .modelSource(ModelSource.fromPath(localDir))
 *         .sessionOptions(opts -> opts.addCUDA(0))
 *         .threshold(0.6f)        // Speech probability threshold
 *         .minSpeechDuration(0.25f)  // Minimum speech segment duration
 *         .minSilenceDuration(0.1f)  // Minimum silence between segments
 *         .build()) {
 *     List<VoiceSegment> segments = vad.detect(audioSamples, 16000);
 * }
 * }</pre>
 *
 * @see VoiceSegment
 * @see VoiceActivityDetector
 */
public class SileroVadDetector implements VoiceActivityDetector {

    private static final String DEFAULT_MODEL_ID = "inference4j/silero-vad";
    private static final int DEFAULT_SAMPLE_RATE = 16000;
    private static final int DEFAULT_WINDOW_SIZE_SAMPLES = 512; // 32ms at 16kHz
    private static final float DEFAULT_THRESHOLD = 0.5f;
    private static final float DEFAULT_MIN_SPEECH_DURATION = 0.25f; // seconds
    private static final float DEFAULT_MIN_SILENCE_DURATION = 0.1f; // seconds

    // Hidden state dimensions (Silero VAD v5 uses combined state tensor)
    private static final int HIDDEN_SIZE = 128;
    private static final int STATE_LAYERS = 2; // h and c combined

    // Context size for Silero VAD v5 (64 samples at 16kHz, 32 at 8kHz)
    private static final int CONTEXT_SIZE_16K = 64;

    private final InferenceSession session;
    private final int targetSampleRate;
    private final int windowSizeSamples;
    private final int contextSize;
    private final float threshold;
    private final float minSpeechDuration;
    private final float minSilenceDuration;


    private SileroVadDetector(InferenceSession session, int targetSampleRate, int windowSizeSamples,
                              int contextSize, float threshold, float minSpeechDuration, float minSilenceDuration) {
        this.session = session;
        this.targetSampleRate = targetSampleRate;
        this.windowSizeSamples = windowSizeSamples;
        this.contextSize = contextSize;
        this.threshold = threshold;
        this.minSpeechDuration = minSpeechDuration;
        this.minSilenceDuration = minSilenceDuration;
    }

    /**
     * Creates a builder for custom configuration.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<VoiceSegment> detect(Path audioPath) {
        AudioData audio = AudioLoader.load(audioPath);
        return detect(audio.samples(), audio.sampleRate());
    }

    @Override
    public List<VoiceSegment> detect(float[] audioData, int sampleRate) {
        // Resample if needed
        float[] samples = AudioProcessor.resample(audioData, sampleRate, targetSampleRate);

        // Get per-frame probabilities
        float[] probabilities = runInference(samples);

        // Convert probabilities to segments
        return extractSegments(probabilities, samples.length);
    }

    /**
     * Returns per-frame speech probabilities for the audio.
     *
     * <p>This is useful for visualization or custom post-processing.
     *
     * @param audioPath path to the audio file (WAV format)
     * @return array of speech probabilities, one per frame
     */
    public float[] probabilities(Path audioPath) {
        AudioData audio = AudioLoader.load(audioPath);
        return probabilities(audio.samples(), audio.sampleRate());
    }

    /**
     * Returns per-frame speech probabilities for raw audio samples.
     *
     * @param audioData  audio samples in [-1.0, 1.0] range
     * @param sampleRate sample rate of the audio in Hz
     * @return array of speech probabilities, one per frame
     */
    public float[] probabilities(float[] audioData, int sampleRate) {
        float[] samples = AudioProcessor.resample(audioData, sampleRate, targetSampleRate);
        return runInference(samples);
    }

    @Override
    public void close() {
        session.close();
    }

    /**
     * Runs the VAD model on audio samples and returns per-frame probabilities.
     */
    private float[] runInference(float[] samples) {
        int numFrames = (int) Math.ceil((double) samples.length / windowSizeSamples);
        float[] probabilities = new float[numFrames];

        // Initialize combined hidden state tensor [2, batch=1, hidden_size]
        float[] state = new float[STATE_LAYERS * 1 * HIDDEN_SIZE];

        // Initialize context buffer with zeros (Silero VAD v5 requires context prepended)
        float[] context = new float[contextSize];

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * windowSizeSamples;
            int end = Math.min(start + windowSizeSamples, samples.length);

            // Extract window with zero-padding if needed
            float[] window = new float[windowSizeSamples];
            int copyLen = end - start;
            System.arraycopy(samples, start, window, 0, copyLen);

            // Create input with context prepended (Silero VAD v5 requirement)
            float[] inputWithContext = new float[contextSize + windowSizeSamples];
            System.arraycopy(context, 0, inputWithContext, 0, contextSize);
            System.arraycopy(window, 0, inputWithContext, contextSize, windowSizeSamples);

            Tensor inputTensor = Tensor.fromFloats(inputWithContext, new long[]{1, contextSize + windowSizeSamples});
            Tensor stateTensor = Tensor.fromFloats(state, new long[]{STATE_LAYERS, 1, HIDDEN_SIZE});
            Tensor srTensor = Tensor.fromLongs(new long[]{targetSampleRate}, new long[]{});

            Map<String, Tensor> inputs = new LinkedHashMap<>();
            inputs.put("input", inputTensor);
            inputs.put("state", stateTensor);
            inputs.put("sr", srTensor);

            Map<String, Tensor> outputs = session.run(inputs);

            // Get output probability
            Tensor outputTensor = outputs.get("output");
            float[] output = outputTensor.toFloats();
            probabilities[frame] = output[0];

            // Update hidden state for next frame
            Tensor stateNTensor = outputs.get("stateN");
            if (stateNTensor != null) {
                state = stateNTensor.toFloats();
            }

            // Update context with the last contextSize samples from the current window
            System.arraycopy(window, windowSizeSamples - contextSize, context, 0, contextSize);
        }

        return probabilities;
    }

    /**
     * Extracts voice segments from frame probabilities.
     *
     * <p>Package-visible for unit testing without an ONNX session.
     */
    List<VoiceSegment> extractSegments(float[] probabilities, int totalSamples) {
        List<VoiceSegment> segments = new ArrayList<>();

        int minSpeechFrames = (int) Math.ceil(minSpeechDuration * targetSampleRate / windowSizeSamples);
        int minSilenceFrames = (int) Math.ceil(minSilenceDuration * targetSampleRate / windowSizeSamples);

        int speechStart = -1;
        int silenceCounter = 0;
        float confidenceSum = 0;
        int speechFrameCount = 0;

        for (int i = 0; i < probabilities.length; i++) {
            boolean isSpeech = probabilities[i] >= threshold;

            if (isSpeech) {
                if (speechStart == -1) {
                    speechStart = i;
                    confidenceSum = 0;
                    speechFrameCount = 0;
                }
                confidenceSum += probabilities[i];
                speechFrameCount++;
                silenceCounter = 0;
            } else if (speechStart != -1) {
                silenceCounter++;

                if (silenceCounter >= minSilenceFrames) {
                    if (speechFrameCount >= minSpeechFrames) {
                        float startTime = frameToTime(speechStart);
                        float endTime = frameToTime(i - silenceCounter + 1);
                        float avgConfidence = confidenceSum / speechFrameCount;
                        segments.add(new VoiceSegment(startTime, endTime, avgConfidence));
                    }
                    speechStart = -1;
                    silenceCounter = 0;
                }
            }
        }

        // Handle segment at end of audio
        if (speechStart != -1 && speechFrameCount >= minSpeechFrames) {
            float startTime = frameToTime(speechStart);
            float endTime = (float) totalSamples / targetSampleRate;
            float avgConfidence = confidenceSum / speechFrameCount;
            segments.add(new VoiceSegment(startTime, endTime, avgConfidence));
        }

        return segments;
    }

    private float frameToTime(int frame) {
        return (float) (frame * windowSizeSamples) / targetSampleRate;
    }

    /**
     * Builder for configuring SileroVadDetector instances.
     */
    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private int sampleRate = DEFAULT_SAMPLE_RATE;
        private int windowSizeSamples = DEFAULT_WINDOW_SIZE_SAMPLES;
        private float threshold = DEFAULT_THRESHOLD;
        private float minSpeechDuration = DEFAULT_MIN_SPEECH_DURATION;
        private float minSilenceDuration = DEFAULT_MIN_SILENCE_DURATION;

        Builder session(InferenceSession session) {
            this.session = session;
            return this;
        }

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder sampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        public Builder windowSizeSamples(int windowSizeSamples) {
            this.windowSizeSamples = windowSizeSamples;
            return this;
        }

        public Builder threshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        public Builder minSpeechDuration(float minSpeechDuration) {
            this.minSpeechDuration = minSpeechDuration;
            return this;
        }

        public Builder minSilenceDuration(float minSilenceDuration) {
            this.minSilenceDuration = minSilenceDuration;
            return this;
        }

        public SileroVadDetector build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id);
                loadFromDirectory(dir);
            }
            int contextSize = (sampleRate == 16000) ? 64 : 32;
            return new SileroVadDetector(session, sampleRate, windowSizeSamples,
                    contextSize, threshold, minSpeechDuration, minSilenceDuration);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                modelPath = dir.resolve("silero_vad.onnx");
            }
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found in: " + dir);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
        }
    }
}
