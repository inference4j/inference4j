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

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class BartSummarizerTest {

    private static final GenerationResult DUMMY_RESULT =
            new GenerationResult("output", 10, 5, Duration.ZERO);

    private GenerationEngine mockEngine() {
        GenerationEngine engine = mock(GenerationEngine.class);
        when(engine.generate(anyString(), any())).thenReturn(DUMMY_RESULT);
        when(engine.generate(anyString())).thenReturn(DUMMY_RESULT);
        return engine;
    }

    @Test
    void summarize_passesTextDirectly_noPrefix() {
        GenerationEngine engine = mockEngine();
        BartSummarizer summarizer = new BartSummarizer(engine);

        summarizer.summarize("The quick brown fox jumps over the lazy dog.", token -> {});

        ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
        verify(engine).generate(promptCaptor.capture(), any());
        assertThat(promptCaptor.getValue())
                .as("BART should pass text directly without any prefix")
                .isEqualTo("The quick brown fox jumps over the lazy dog.");
    }

    @Test
    void generate_delegatesToEngine() {
        GenerationEngine engine = mockEngine();
        BartSummarizer summarizer = new BartSummarizer(engine);

        GenerationResult result = summarizer.generate("Hello");

        verify(engine).generate("Hello");
        assertThat(result.text()).isEqualTo("output");
    }

    @Test
    void distilBartCnn_preset_returnsBuilder() {
        BartSummarizer.Builder builder = BartSummarizer.distilBartCnn();
        assertThat(builder).isNotNull();
    }

    @Test
    void bartLargeCnn_preset_returnsBuilder() {
        BartSummarizer.Builder builder = BartSummarizer.bartLargeCnn();
        assertThat(builder).isNotNull();
    }

    @Test
    void builder_noModelIdOrSource_throws() {
        assertThatThrownBy(() -> BartSummarizer.builder().build())
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> {
                    assertThat(ex.getMessage()).contains("modelId");
                    assertThat(ex.getMessage()).contains("modelSource");
                });
    }
}
