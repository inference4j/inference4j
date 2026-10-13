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

/**
 * Live transcription of a continuous audio stream, such as a microphone or a broadcast.
 *
 * <p>Audio goes in as it arrives, in pieces of any size. A voice activity detector splits it into lines at
 * pauses; while a line is open its text is updated at regular intervals, and when the speaker pauses the
 * line is completed. Results arrive through a {@link TranscriptListener}.
 *
 * <p>All work happens inside {@link #addAudio(float[], int)}, on the caller's thread, and listener events
 * fire on that thread too. An update can take tens of milliseconds, so when capturing from a microphone,
 * read audio on one thread and hand it to the stream on another, through a queue.
 *
 * <p>Not thread-safe: use one stream from one thread at a time.
 *
 * @see MoonshineRecognizer#stream(SileroVadDetector)
 */
public interface TranscriptionStream extends AutoCloseable {

    /**
     * Adds audio to the stream.
     *
     * @param samples    mono samples in [-1.0, 1.0]
     * @param sampleRate the sample rate in Hz; must be the same for every call
     * @throws IllegalArgumentException if the sample rate differs from earlier calls
     */
    void addAudio(float[] samples, int sampleRate);

    /**
     * Completes the open line, if any, as if the speaker had paused. Use it when the audio source stops,
     * for example when the user releases a push-to-talk button.
     */
    void flush();

    /** Completes the open line, if any, and releases the stream's resources. */
    @Override
    void close();
}
