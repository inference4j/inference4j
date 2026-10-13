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

/*
 * Portions of this file are ported from moonshine-ai/moonshine (core/voice-activity-detector.cpp and
 * core/transcriber.cpp), Copyright (c) 2025 Useful Sensors, Inc. (dba Moonshine AI), licensed under the
 * MIT License. See the NOTICE file for the full license text.
 */

package io.github.inference4j.audio;

import io.github.inference4j.tokenizer.TokenDecoder;

import java.time.Duration;
import java.util.Arrays;

/**
 * Live transcription with Moonshine, segmented into lines by Silero VAD.
 *
 * <p>Every 32 ms window of audio is scored by the VAD. Speech is when the average probability over the
 * last 0.5 s is above the detector's threshold. A line starts with 0.5 s of audio from before the speech
 * began, so the first word is not clipped. While the line is open, its audio is encoded as it arrives and
 * the line is re-decoded once per update interval, reusing the previous text as a speculative draft. When
 * the average drops below the threshold, the line is encoded and decoded one last time and completed.
 * Past two thirds of the maximum line length, the speech score fades linearly to zero at the maximum, so
 * a speaker who never pauses still gets lines that end at a natural dip.
 */
final class MoonshineTranscriptionStream implements TranscriptionStream {

    private static final int SAMPLE_RATE = MoonshineModel.SAMPLE_RATE;

    /** Window over which speech probabilities are averaged. */
    private static final double SMOOTHING_SECONDS = 0.5;

    /** Audio kept from before speech starts, so the start of the first word is in the line. */
    private static final int LOOK_BEHIND_SAMPLES = 8192;

    private final MoonshineModel model;
    private final TokenDecoder decoder;
    private final Object lock;
    private final SileroVadDetector.Scorer scorer;
    private final float threshold;
    private final int window;
    private final int updateSamples;
    private final int maxLineSamples;
    private final TranscriptListener listener;

    private StreamingResampler resampler;
    private int inputSampleRate;

    // VAD state
    private final float[] probabilities;
    private int probabilityIndex;
    private final float[] hop;
    private int hopLength;
    private final float[] lookBehind = new float[LOOK_BEHIND_SAMPLES];
    private long samplesProcessed;
    private boolean speaking;

    // Open line
    private MoonshineModel.Line line;
    private long lineId = -1;
    private long lineStartSample;
    private int lineSamples;
    private int samplesSinceUpdate;
    private int[] draft = new int[0];
    private String text = "";

    private boolean closed;

    MoonshineTranscriptionStream(MoonshineModel model, TokenDecoder decoder, Object lock, SileroVadDetector vad,
                                 Duration updateInterval, Duration maxLineDuration, TranscriptListener listener) {
        if (vad.sampleRate() != SAMPLE_RATE) {
            throw new IllegalArgumentException("The VAD must run at " + SAMPLE_RATE + " Hz, got " + vad.sampleRate());
        }
        this.model = model;
        this.decoder = decoder;
        this.lock = lock;
        this.scorer = vad.newScorer();
        this.threshold = vad.threshold();
        this.window = vad.windowSizeSamples();
        this.updateSamples = toSamples(updateInterval);
        this.maxLineSamples = toSamples(maxLineDuration);
        this.listener = listener;
        this.probabilities = new float[(int) Math.ceil(SMOOTHING_SECONDS * SAMPLE_RATE / window)];
        this.hop = new float[window];
    }

    private static int toSamples(Duration duration) {
        return (int) (duration.toNanos() * SAMPLE_RATE / 1_000_000_000L);
    }

    @Override
    public void addAudio(float[] samples, int sampleRate) {
        ensureOpen();
        if (resampler == null && inputSampleRate == 0) {
            inputSampleRate = sampleRate;
            resampler = sampleRate == SAMPLE_RATE ? null : new StreamingResampler(sampleRate, SAMPLE_RATE);
        } else if (sampleRate != inputSampleRate) {
            throw new IllegalArgumentException("Sample rate changed from " + inputSampleRate + " to " + sampleRate);
        }
        float[] audio = resampler == null ? samples : resampler.resample(samples);
        synchronized (lock) {
            int offset = 0;
            while (offset < audio.length) {
                int take = Math.min(window - hopLength, audio.length - offset);
                System.arraycopy(audio, offset, hop, hopLength, take);
                hopLength += take;
                offset += take;
                if (hopLength == window) {
                    processHop();
                    hopLength = 0;
                }
            }
        }
    }

    private void processHop() {
        samplesProcessed += window;
        System.arraycopy(lookBehind, window, lookBehind, 0, LOOK_BEHIND_SAMPLES - window);
        System.arraycopy(hop, 0, lookBehind, LOOK_BEHIND_SAMPLES - window, window);

        probabilities[probabilityIndex] = scorer.score(hop);
        probabilityIndex = (probabilityIndex + 1) % probabilities.length;
        float smoothed = 0;
        for (float p : probabilities) {
            smoothed += p;
        }
        smoothed /= probabilities.length;

        int fadeStart = maxLineSamples * 2 / 3;
        if (line != null && lineSamples > fadeStart) {
            smoothed *= Math.max(0f, (float) (maxLineSamples - lineSamples) / (maxLineSamples - fadeStart));
        }
        boolean voice = smoothed > threshold;

        if (voice && !speaking) {
            int behind = (int) Math.min(LOOK_BEHIND_SAMPLES, samplesProcessed);
            startLine(Arrays.copyOfRange(lookBehind, LOOK_BEHIND_SAMPLES - behind, LOOK_BEHIND_SAMPLES));
        } else if (voice) {
            appendToLine(hop);
            if (samplesSinceUpdate >= updateSamples) {
                updateLine();
            }
        } else if (speaking) {
            appendToLine(hop);
            completeLine();
        }
        speaking = voice;
    }

    private void startLine(float[] audio) {
        line = model.newLine();
        lineId++;
        lineStartSample = samplesProcessed - audio.length;
        lineSamples = 0;
        samplesSinceUpdate = 0;
        draft = new int[0];
        text = "";
        appendToLine(audio);
        listener.onLineStarted(snapshot(false));
    }

    private void appendToLine(float[] audio) {
        line.addAudio(audio);
        lineSamples += audio.length;
        samplesSinceUpdate += audio.length;
    }

    private void updateLine() {
        samplesSinceUpdate = 0;
        line.encode(false);
        if (decode()) {
            listener.onLineTextChanged(snapshot(false));
        }
    }

    private void completeLine() {
        line.encode(true);
        decode();
        TranscriptLine completed = snapshot(true);
        line.close();
        line = null;
        speaking = false;
        listener.onLineCompleted(completed);
    }

    /** Decodes the open line; returns whether its text changed. */
    private boolean decode() {
        draft = line.decode(MoonshineRecognizer.maxTokens(lineSamples), draft);
        String decoded = decoder.decode(draft).strip();
        boolean changed = !decoded.equals(text);
        text = decoded;
        return changed;
    }

    private TranscriptLine snapshot(boolean completed) {
        return new TranscriptLine(lineId, text, toDuration(lineStartSample),
                toDuration(lineStartSample + lineSamples), completed);
    }

    private static Duration toDuration(long samples) {
        return Duration.ofNanos(samples * 1_000_000_000L / SAMPLE_RATE);
    }

    @Override
    public void flush() {
        ensureOpen();
        synchronized (lock) {
            if (line != null) {
                if (hopLength > 0) {
                    appendToLine(Arrays.copyOf(hop, hopLength));
                    samplesProcessed += hopLength;
                    hopLength = 0;
                }
                completeLine();
            }
            // The next line starts afresh, as after a pause
            Arrays.fill(probabilities, 0f);
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        flush();
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("The transcription stream is closed");
        }
    }
}
