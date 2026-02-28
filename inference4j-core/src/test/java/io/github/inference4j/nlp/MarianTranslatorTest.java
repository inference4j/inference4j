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

package io.github.inference4j.nlp;

import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.generation.GenerationEngine;
import io.github.inference4j.generation.GenerationResult;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class MarianTranslatorTest {

    private static final GenerationResult DUMMY_RESULT =
            new GenerationResult("output", 10, 5, Duration.ZERO);

    private GenerationEngine mockEngine() {
        GenerationEngine engine = mock(GenerationEngine.class);
        when(engine.generate(anyString(), any())).thenReturn(DUMMY_RESULT);
        when(engine.generate(anyString())).thenReturn(DUMMY_RESULT);
        return engine;
    }

    @Test
    void translate_passesTextDirectly() {
        GenerationEngine engine = mockEngine();
        MarianTranslator translator = new MarianTranslator(engine);

        translator.translate("Hello, how are you?", token -> {});

        ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
        verify(engine).generate(promptCaptor.capture(), any());
        assertEquals("Hello, how are you?", promptCaptor.getValue(),
                "MarianMT should pass text directly without any prefix");
    }

    @Test
    void translate_withLanguages_throwsUnsupported() {
        GenerationEngine engine = mockEngine();
        MarianTranslator translator = new MarianTranslator(engine);

        UnsupportedOperationException ex = assertThrows(UnsupportedOperationException.class, () ->
                translator.translate("Hello", Language.EN, Language.DE, token -> {}));
        assertTrue(ex.getMessage().contains("fixed language pair"),
                "Message should mention fixed language pair but was: " + ex.getMessage());
    }

    @Test
    void generate_delegatesToEngine() {
        GenerationEngine engine = mockEngine();
        MarianTranslator translator = new MarianTranslator(engine);

        GenerationResult result = translator.generate("Hello");

        verify(engine).generate("Hello");
        assertEquals("output", result.text());
    }

    @Test
    void builder_noModelId_throws() {
        ModelLoadException ex = assertThrows(ModelLoadException.class, () ->
                MarianTranslator.builder().build());
        assertTrue(ex.getMessage().contains("modelId"));
        assertTrue(ex.getMessage().contains("modelSource"));
    }

    @Test
    void builder_fluentApi_acceptsModelId() {
        MarianTranslator.Builder builder = MarianTranslator.builder()
                .modelId("Helsinki-NLP/opus-mt-en-de");
        assertNotNull(builder);
    }
}
