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

import java.time.Duration;

/**
 * One line of a live transcript: a stretch of speech between pauses.
 *
 * <p>While the speaker is talking, a line is open and its text may change with every update. Once
 * {@link #completed()} is {@code true}, the line never changes again.
 *
 * @param id        identifies the line within its stream; lines are numbered from 0 in the order they start
 * @param text      the transcribed text so far
 * @param start     where the line starts, measured from the start of the stream
 * @param end       where the line ends so far, measured from the start of the stream
 * @param completed whether the line is final
 */
public record TranscriptLine(long id, String text, Duration start, Duration end, boolean completed) {
}
