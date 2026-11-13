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

import java.util.List;

/**
 * Result of a speech-to-text transcription.
 *
 * <p>The {@link #segments} list contains timed segments with start and end times in seconds, when
 * timing is available. Recognizers on their own return plain text with an empty list;
 * {@link SegmentedSpeechRecognizer} adds one segment per piece of audio it transcribes.
 *
 * @param text the full transcribed text
 * @param segments timed segments of the transcription, empty if timestamps are not available
 */
public record Transcription(String text, List<Segment> segments) {

	/**
	 * Creates a transcription with no timed segments.
	 *
	 * <p>This convenience constructor is used by models that produce only plain text
	 * without timestamp information.
	 *
	 * @param text the transcribed text
	 */
	public Transcription(String text) {
		this(text, List.of());
	}

	/**
	 * A timed segment of a transcription.
	 *
	 * @param text the text content of this segment
	 * @param startTime the start time in seconds
	 * @param endTime the end time in seconds
	 */
	public record Segment(String text, float startTime, float endTime) {
	}

}
