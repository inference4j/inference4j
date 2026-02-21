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

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SileroVadDetectorModelTest {

    private static Path extractSpeechFixture() throws IOException {
        Path tempFile = Files.createTempFile("speech-fixture-", ".wav");
        tempFile.toFile().deleteOnExit();
        try (InputStream is = SileroVadDetectorModelTest.class.getResourceAsStream("/fixtures/speech.wav")) {
            Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
        }
        return tempFile;
    }

    @Test
    void detect_speechWav_findsVoiceSegments() throws IOException {
        try (var vad = io.github.inference4j.audio.SileroVadDetector.builder().build()) {
            List<VoiceSegment> segments = vad.detect(extractSpeechFixture());

            assertFalse(segments.isEmpty(), "Should detect at least one voice segment in speech audio");
        }
    }

    @Test
    void detect_speechWav_segmentsHaveValidTimestamps() throws IOException {
        try (var vad = SileroVadDetector.builder().build()) {
            List<VoiceSegment> segments = vad.detect(extractSpeechFixture());

            for (VoiceSegment segment : segments) {
                assertTrue(segment.start() >= 0f,
                        "Segment start should be >= 0, got: " + segment.start());
                assertTrue(segment.end() > segment.start(),
                        "Segment end should be > start: start=" + segment.start() + " end=" + segment.end());
                assertTrue(segment.duration() > 0f,
                        "Segment duration should be positive, got: " + segment.duration());
                assertTrue(segment.confidence() > 0f && segment.confidence() <= 1f,
                        "Segment confidence should be (0, 1], got: " + segment.confidence());
            }
        }
    }
}
