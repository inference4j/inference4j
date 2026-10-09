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

import io.github.inference4j.nlp.PunctCapSegModel;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static org.assertj.core.api.Assertions.assertThat;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PunctuatedSpeechRecognizerModelTest {

    private SpeechRecognizer recognizer;
    private Path speechFixture;

    @BeforeAll
    void setUp() throws IOException {
        recognizer = new PunctuatedSpeechRecognizer(
                Wav2Vec2Recognizer.builder().build(),
                PunctCapSegModel.builder().build());
        speechFixture = Files.createTempFile("speech-fixture-", ".wav");
        speechFixture.toFile().deleteOnExit();
        try (InputStream is = PunctuatedSpeechRecognizerModelTest.class.getResourceAsStream("/fixtures/speech.wav")) {
            Files.copy(is, speechFixture, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    @AfterAll
    void tearDown() {
        if (recognizer != null) recognizer.close();
    }

    @Test
    void transcriptIsCasedAndPunctuated() {
        String text = recognizer.transcribe(speechFixture).text();

        // Wav2Vec2 alone returns all upper-case text with no punctuation
        assertThat(text).matches(".*[a-z].*");
        assertThat(text.charAt(0)).isUpperCase();
        assertThat(text).matches(".*[.?]$");
    }
}
