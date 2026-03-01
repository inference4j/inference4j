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

import static org.assertj.core.api.Assertions.*;

class TranscriptionTest {

	@Test
	void convenienceConstructor_createsEmptySegments() {
		Transcription transcription = new Transcription("hello world");

		assertThat(transcription.text()).isEqualTo("hello world");
		assertThat(transcription.segments()).isNotNull();
		assertThat(transcription.segments()).isEmpty();
	}

	@Test
	void fullConstructor_preservesSegments() {
		var seg1 = new Transcription.Segment("hello", 0.0f, 1.5f);
		var seg2 = new Transcription.Segment("world", 1.5f, 3.0f);

		Transcription transcription = new Transcription("hello world", List.of(seg1, seg2));

		assertThat(transcription.text()).isEqualTo("hello world");
		assertThat(transcription.segments()).hasSize(2);
		assertThat(transcription.segments().get(0)).isEqualTo(seg1);
		assertThat(transcription.segments().get(1)).isEqualTo(seg2);
	}

	@Test
	void segment_recordEquality() {
		var seg1 = new Transcription.Segment("hello", 0.0f, 1.5f);
		var seg2 = new Transcription.Segment("hello", 0.0f, 1.5f);

		assertThat(seg2).isEqualTo(seg1);
		assertThat(seg2.hashCode()).isEqualTo(seg1.hashCode());
	}

	@Test
	void convenienceConstructor_segmentsListIsImmutable() {
		Transcription transcription = new Transcription("hello world");

		assertThatThrownBy(() -> {
			transcription.segments().add(new Transcription.Segment("nope", 0.0f, 1.0f));
		}).isInstanceOf(UnsupportedOperationException.class);
	}

}
