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
 *
 * <p>Model work for each 32 ms hop holds the recognizer open (see {@link MoonshineGuard}), so streams of one
 * recognizer run in parallel and the recognizer cannot be closed mid-hop. Listener events fire after the hop,
 * with no lock held and the stream's state final, so a listener may flush or close this stream, or close the
 * recognizer.
 */
final class MoonshineTranscriptionStream implements TranscriptionStream {

    private static final int SAMPLE_RATE = MoonshineModel.SAMPLE_RATE;

    /** Window over which speech probabilities are averaged. */
    private static final double SMOOTHING_SECONDS = 0.5;

    /** Audio kept from before speech starts, so the start of the first word is in the line. */
    private static final int LOOK_BEHIND_SAMPLES = 8192;

    private final MoonshineModel model;
    private final TokenDecoder decoder;
    private final MoonshineGuard guard;
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

    MoonshineTranscriptionStream(MoonshineModel model, TokenDecoder decoder, MoonshineGuard guard,
                                 SileroVadDetector vad,
                                 Duration updateInterval, Duration maxLineDuration, TranscriptListener listener) {
        if (vad.sampleRate() != SAMPLE_RATE) {
            throw new IllegalArgumentException("The VAD must run at " + SAMPLE_RATE + " Hz, got " + vad.sampleRate());
        }
        this.model = model;
        this.decoder = decoder;
        this.guard = guard;
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
        if (guard.isClosed()) {
            throw new IllegalStateException("The speech recognizer was closed");
        }
        float[] audio = resampler == null ? samples : resampler.resample(samples);
        int offset = 0;
        while (offset < audio.length) {
            int take = Math.min(window - hopLength, audio.length - offset);
            System.arraycopy(audio, offset, hop, hopLength, take);
            hopLength += take;
            offset += take;
            if (hopLength == window) {
                Event event;
                guard.acquire();
                try {
                    // The hop is used up even if processing it fails, so it is never counted twice
                    hopLength = 0;
                    event = processHop();
                } finally {
                    guard.release();
                }
                fire(event);
                if (closed) {
                    // The listener closed this stream
                    return;
                }
            }
        }
    }

    /** Scores one hop and moves the open line along; returns the event it produced, or {@code null}. */
    private Event processHop() {
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

        Event event = null;
        if (voice && !speaking) {
            int behind = (int) Math.min(LOOK_BEHIND_SAMPLES, samplesProcessed);
            event = startLine(Arrays.copyOfRange(lookBehind, LOOK_BEHIND_SAMPLES - behind, LOOK_BEHIND_SAMPLES));
        } else if (voice) {
            appendToLine(hop);
            if (samplesSinceUpdate >= updateSamples) {
                event = updateLine();
            }
        } else if (speaking) {
            appendToLine(hop);
            event = completeLine();
        }
        speaking = voice;
        return event;
    }

    private Event startLine(float[] audio) {
        line = model.newLine();
        lineId++;
        lineStartSample = samplesProcessed - audio.length;
        lineSamples = 0;
        samplesSinceUpdate = 0;
        draft = new int[0];
        text = "";
        appendToLine(audio);
        return new Event(EventKind.STARTED, snapshot(false));
    }

    private void appendToLine(float[] audio) {
        line.addAudio(audio);
        lineSamples += audio.length;
        samplesSinceUpdate += audio.length;
    }

    private Event updateLine() {
        samplesSinceUpdate = 0;
        line.encode(false);
        return decode() ? new Event(EventKind.TEXT_CHANGED, snapshot(false)) : null;
    }

    private Event completeLine() {
        line.encode(true);
        decode();
        TranscriptLine completed = snapshot(true);
        line.close();
        line = null;
        speaking = false;
        return new Event(EventKind.COMPLETED, completed);
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
        Event event;
        guard.acquire();
        try {
            event = finishLine();
        } finally {
            guard.release();
        }
        fire(event);
    }

    /** Completes the open line, if any, with the audio of the partial hop. Call while holding the model. */
    private Event finishLine() {
        Event event = null;
        if (line != null) {
            if (hopLength > 0) {
                appendToLine(Arrays.copyOf(hop, hopLength));
                samplesProcessed += hopLength;
                hopLength = 0;
            }
            event = completeLine();
        }
        // The next line starts afresh, as after a pause
        Arrays.fill(probabilities, 0f);
        return event;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        Event event = null;
        try {
            if (guard.acquireIfOpen()) {
                try {
                    event = finishLine();
                } finally {
                    guard.release();
                }
            }
        } finally {
            if (line != null) {
                // The recognizer is closed, or finishing the line failed: free it without a last decode
                line.close();
                line = null;
            }
            closed = true;
        }
        fire(event);
    }

    private void fire(Event event) {
        if (event == null) {
            return;
        }
        switch (event.kind()) {
            case STARTED -> listener.onLineStarted(event.line());
            case TEXT_CHANGED -> listener.onLineTextChanged(event.line());
            case COMPLETED -> listener.onLineCompleted(event.line());
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("The transcription stream is closed");
        }
    }

    private enum EventKind { STARTED, TEXT_CHANGED, COMPLETED }

    /** A listener event produced while holding the model, fired after releasing it. */
    private record Event(EventKind kind, TranscriptLine line) {
    }
}
