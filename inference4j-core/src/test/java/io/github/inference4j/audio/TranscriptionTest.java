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

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TranscriptionTest {

	@Test
	void convenienceConstructor_createsEmptySegments() {
		Transcription transcription = new Transcription("hello world");

		assertEquals("hello world", transcription.text());
		assertNotNull(transcription.segments());
		assertTrue(transcription.segments().isEmpty());
	}

	@Test
	void fullConstructor_preservesSegments() {
		var seg1 = new Transcription.Segment("hello", 0.0f, 1.5f);
		var seg2 = new Transcription.Segment("world", 1.5f, 3.0f);

		Transcription transcription = new Transcription("hello world", List.of(seg1, seg2));

		assertEquals("hello world", transcription.text());
		assertEquals(2, transcription.segments().size());
		assertEquals(seg1, transcription.segments().get(0));
		assertEquals(seg2, transcription.segments().get(1));
	}

	@Test
	void segment_recordEquality() {
		var seg1 = new Transcription.Segment("hello", 0.0f, 1.5f);
		var seg2 = new Transcription.Segment("hello", 0.0f, 1.5f);

		assertEquals(seg1, seg2);
		assertEquals(seg1.hashCode(), seg2.hashCode());
	}

	@Test
	void convenienceConstructor_segmentsListIsImmutable() {
		Transcription transcription = new Transcription("hello world");

		assertThrows(UnsupportedOperationException.class, () -> {
			transcription.segments().add(new Transcription.Segment("nope", 0.0f, 1.0f));
		});
	}

}
