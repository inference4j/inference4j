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
 * Receives the events of a {@link TranscriptionStream}. Every method does nothing by default, so implement
 * only the ones you need.
 *
 * <p>Events fire on the thread that calls {@link TranscriptionStream#addAudio(float[], int)},
 * {@link TranscriptionStream#flush()} or {@link TranscriptionStream#close()}. For one line they always come
 * in this order: {@code onLineStarted}, any number of {@code onLineTextChanged}, then
 * {@code onLineCompleted}. Only one line is open at a time.
 */
public interface TranscriptListener {

    /** A pause ended and speech began; the line has no text yet. */
    default void onLineStarted(TranscriptLine line) {
    }

    /** The open line's text changed. */
    default void onLineTextChanged(TranscriptLine line) {
    }

    /** The speaker paused, or the line reached its maximum length; this is the line's final text. */
    default void onLineCompleted(TranscriptLine line) {
    }
}
