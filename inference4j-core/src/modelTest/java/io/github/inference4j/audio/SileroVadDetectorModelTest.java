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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class SileroVadDetectorModelTest {

    private SileroVadDetector vad;
    private Path speechFixture;

    @BeforeAll
    void setUp() throws IOException {
        vad = SileroVadDetector.builder().build();
        speechFixture = Files.createTempFile("speech-fixture-", ".wav");
        speechFixture.toFile().deleteOnExit();
        try (InputStream is = SileroVadDetectorModelTest.class.getResourceAsStream("/fixtures/speech.wav")) {
            Files.copy(is, speechFixture, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    @AfterAll
    void tearDown() throws Exception {
        if (vad != null) vad.close();
    }

    @Test
    void detect_speechWav_findsVoiceSegments() {
        List<VoiceSegment> segments = vad.detect(speechFixture);

        assertThat(segments.isEmpty()).as("Should detect at least one voice segment in speech audio").isFalse();
    }

    @Test
    void detect_speechWav_segmentsHaveValidTimestamps() {
        List<VoiceSegment> segments = vad.detect(speechFixture);

        for (VoiceSegment segment : segments) {
            assertThat(segment.start() >= 0f).as("Segment start should be >= 0, got: " + segment.start()).isTrue();
            assertThat(segment.end() > segment.start()).as("Segment end should be > start: start=" + segment.start() + " end=" + segment.end()).isTrue();
            assertThat(segment.duration() > 0f).as("Segment duration should be positive, got: " + segment.duration()).isTrue();
            assertThat(segment.confidence() > 0f && segment.confidence() <= 1f).as("Segment confidence should be (0, 1], got: " + segment.confidence()).isTrue();
        }
    }
}
