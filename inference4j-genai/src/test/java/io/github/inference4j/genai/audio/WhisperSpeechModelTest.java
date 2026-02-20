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
package io.github.inference4j.genai.audio;

import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.MultiModalProcessor;
import io.github.inference4j.audio.Transcription;
import io.github.inference4j.genai.audio.WhisperSpeechModel;
import io.github.inference4j.genai.audio.WhisperTask;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class WhisperSpeechModelTest {

    @Test
    void parseOutput_stripsWhitespaceAndWrapsInTranscription() {
        Model model = mock(Model.class);
        MultiModalProcessor processor = mock(MultiModalProcessor.class);

        WhisperSpeechModel whisper = new WhisperSpeechModel(
                model, processor, "en", WhisperTask.TRANSCRIBE, 448, 1.0, 0, 0.0);

        Transcription result = whisper.parseOutput(
                "  Hello world  ", null, 5, 100);

        assertEquals("Hello world", result.text());
        assertTrue(result.segments().isEmpty());
    }

    @Test
    void closeReleasesProcessorAndModel() {
        Model model = mock(Model.class);
        MultiModalProcessor processor = mock(MultiModalProcessor.class);

        WhisperSpeechModel whisper = new WhisperSpeechModel(
                model, processor, "en", WhisperTask.TRANSCRIBE, 448, 1.0, 0, 0.0);
        whisper.close();

        verify(processor).close();
        verify(model).close();
    }

    @Test
    void buildPrompt_transcribeEnglish() {
        Model model = mock(Model.class);
        MultiModalProcessor processor = mock(MultiModalProcessor.class);

        WhisperSpeechModel whisper = new WhisperSpeechModel(
                model, processor, "en", WhisperTask.TRANSCRIBE, 448, 1.0, 0, 0.0);

        assertEquals("<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
                whisper.buildPrompt());
    }

    @Test
    void buildPrompt_translateFrench() {
        Model model = mock(Model.class);
        MultiModalProcessor processor = mock(MultiModalProcessor.class);

        WhisperSpeechModel whisper = new WhisperSpeechModel(
                model, processor, "fr", WhisperTask.TRANSLATE, 448, 1.0, 0, 0.0);

        assertEquals("<|startoftranscript|><|fr|><|translate|><|notimestamps|>",
                whisper.buildPrompt());
    }

    @Test
    void builderRequiresModelId() {
        assertThrows(IllegalStateException.class, () ->
                WhisperSpeechModel.builder().build());
    }

    @Test
    void builderDefaultsToEnglishTranscription() {
        Model model = mock(Model.class);
        MultiModalProcessor processor = mock(MultiModalProcessor.class);

        WhisperSpeechModel.Builder builder = WhisperSpeechModel.builder();
        builder.model = model;
        builder.processor = processor;
        WhisperSpeechModel whisper = builder.build();

        assertEquals("<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
                whisper.buildPrompt());
    }
}
