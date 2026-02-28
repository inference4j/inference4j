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
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class FlanT5TextGeneratorTest {

    private static final GenerationResult DUMMY_RESULT =
            new GenerationResult("output", 10, 5, Duration.ZERO);

    private GenerationEngine mockEngine() {
        GenerationEngine engine = mock(GenerationEngine.class);
        when(engine.generate(anyString(), any())).thenReturn(DUMMY_RESULT);
        when(engine.generate(anyString())).thenReturn(DUMMY_RESULT);
        return engine;
    }

    @Test
    void summarize_prependsSummarizePrefix() {
        GenerationEngine engine = mockEngine();
        FlanT5TextGenerator generator = new FlanT5TextGenerator(engine);

        generator.summarize("The quick brown fox jumps over the lazy dog.", token -> {});

        ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
        verify(engine).generate(promptCaptor.capture(), any());
        assertTrue(promptCaptor.getValue().startsWith("summarize: "),
                "Prompt should start with 'summarize: ' but was: " + promptCaptor.getValue());
        assertTrue(promptCaptor.getValue().contains("The quick brown fox"));
    }

    @Test
    void translate_prependsTranslatePrefix() {
        GenerationEngine engine = mockEngine();
        FlanT5TextGenerator generator = new FlanT5TextGenerator(engine);

        generator.translate("Hello world", Language.EN, Language.FR, token -> {});

        ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
        verify(engine).generate(promptCaptor.capture(), any());
        assertTrue(promptCaptor.getValue().startsWith("translate English to French: "),
                "Prompt should start with 'translate English to French: ' but was: " + promptCaptor.getValue());
        assertTrue(promptCaptor.getValue().contains("Hello world"));
    }

    @Test
    void translate_noArgs_throwsUnsupported() {
        GenerationEngine engine = mockEngine();
        FlanT5TextGenerator generator = new FlanT5TextGenerator(engine);

        assertThrows(UnsupportedOperationException.class, () ->
                generator.translate("Hello world", token -> {}));
    }

    @Test
    void correct_prependsGrammarPrefix() {
        GenerationEngine engine = mockEngine();
        FlanT5TextGenerator generator = new FlanT5TextGenerator(engine);

        generator.correct("She don't likes the weathers today", token -> {});

        ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
        verify(engine).generate(promptCaptor.capture(), any());
        assertTrue(promptCaptor.getValue().startsWith("correct grammar: "),
                "Prompt should start with 'correct grammar: ' but was: " + promptCaptor.getValue());
        assertTrue(promptCaptor.getValue().contains("She don't likes the weathers today"));
    }

    @Test
    void summarize_withStreaming_delegatesToEngine() {
        GenerationEngine engine = mockEngine();
        FlanT5TextGenerator generator = new FlanT5TextGenerator(engine);
        Consumer<String> listener = token -> {};

        GenerationResult result = generator.summarize("Some long article text.", listener);

        verify(engine).generate(anyString(), eq(listener));
        assertNotNull(result);
        assertEquals("output", result.text());
    }

    @Test
    void generate_delegatesToEngine() {
        GenerationEngine engine = mockEngine();
        FlanT5TextGenerator generator = new FlanT5TextGenerator(engine);

        GenerationResult result = generator.generate("Hello");

        verify(engine).generate("Hello");
        assertEquals("output", result.text());
    }

    @Test
    void flanT5Small_preset_returnsBuilder() {
        FlanT5TextGenerator.Builder builder = FlanT5TextGenerator.flanT5Small();
        assertNotNull(builder);
    }

    @Test
    void flanT5Base_preset_returnsBuilder() {
        FlanT5TextGenerator.Builder builder = FlanT5TextGenerator.flanT5Base();
        assertNotNull(builder);
    }

    @Test
    void flanT5Large_preset_returnsBuilder() {
        FlanT5TextGenerator.Builder builder = FlanT5TextGenerator.flanT5Large();
        assertNotNull(builder);
    }

    @Test
    void builder_noModelIdOrSource_throws() {
        ModelLoadException ex = assertThrows(ModelLoadException.class, () ->
                FlanT5TextGenerator.builder().build());
        assertTrue(ex.getMessage().contains("modelId"));
        assertTrue(ex.getMessage().contains("modelSource"));
    }
}
