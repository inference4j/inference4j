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
 * A detected voice activity segment from audio.
 *
 * <p>Represents a contiguous region of speech detected by a Voice Activity
 * Detector (VAD). Times are in seconds relative to the start of the audio.
 *
 * @param start      start time in seconds
 * @param end        end time in seconds
 * @param confidence average confidence score for the segment (0.0 to 1.0)
 */
public record VoiceSegment(float start, float end, float confidence) {

    /**
     * Returns the duration of this segment in seconds.
     */
    public float duration() {
        return end - start;
    }
}

