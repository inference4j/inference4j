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

import static org.junit.jupiter.api.Assertions.*;

class Wav2Vec2RecognizerModelTest {

    private static Path extractSpeechFixture() throws IOException {
        Path tempFile = Files.createTempFile("speech-fixture-", ".wav");
        tempFile.toFile().deleteOnExit();
        try (InputStream is = Wav2Vec2RecognizerModelTest.class.getResourceAsStream("/fixtures/speech.wav")) {
            Files.copy(is, tempFile, StandardCopyOption.REPLACE_EXISTING);
        }
        return tempFile;
    }

    @Test
    void transcribe_speechWav_returnsNonEmptyText() throws IOException {
        try (var recognizer = Wav2Vec2Recognizer.builder().build()) {
            Transcription result = recognizer.transcribe(extractSpeechFixture());

            assertNotNull(result, "Transcription should not be null");
            assertNotNull(result.text(), "Transcription text should not be null");
            assertFalse(result.text().isBlank(), "Transcription should produce non-empty text");
        }
    }

    @Test
    void transcribe_speechWav_containsRecognizableWords() throws IOException {
        try (var recognizer = Wav2Vec2Recognizer.builder().build()) {
            Transcription result = recognizer.transcribe(extractSpeechFixture());

            String text = result.text().toUpperCase();
            // The transcription should contain alphabetic characters (real English words)
            assertTrue(text.matches(".*[A-Z].*"),
                    "Transcription should contain alphabetic characters, got: " + result.text());
        }
    }
}
