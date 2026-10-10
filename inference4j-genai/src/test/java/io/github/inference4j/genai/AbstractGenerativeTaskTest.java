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
package io.github.inference4j.genai;

import ai.onnxruntime.genai.GenAIException;
import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.TokenizerStream;
import io.github.inference4j.generation.GenerationResult;
import org.junit.jupiter.api.Test;
import org.mockito.MockedConstruction;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockConstruction;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AbstractGenerativeTaskTest {

    @Test
    void closeReleasesModel() throws Exception {
        Model model = mock(Model.class);

        var task = new TestGenerativeTask(model);
        task.close();

        verify(model).close();
    }

    @Test
    void closeCallsCloseResources() throws Exception {
        Model model = mock(Model.class);

        var task = new TestGenerativeTask(model);
        task.close();

        assertThat(task.closeResourcesCalled)
                .as("closeResources() should be called during close()").isTrue();
        verify(model).close();
    }

    @Test
    void endOfTextStepDoesNotRepeatTheLastToken() throws Exception {
        TokenizerStream stream = mock(TokenizerStream.class);
        when(stream.decode(10)).thenReturn(" country");
        when(stream.decode(13)).thenReturn(".");
        List<String> streamed = new ArrayList<>();

        // Prompt of 4 tokens, then " country", ".", and end-of-text, which genai does not append
        try (MockedConstruction<Generator> ignored = mockConstruction(Generator.class, (generator, context) -> {
            when(generator.isDone()).thenReturn(false, false, false, true);
            when(generator.tokenCount()).thenReturn(4L, 5L, 6L, 6L);
            when(generator.getLastTokenInSequence(0)).thenReturn(10, 13, 13);
        })) {
            GenerationResult result = new TestGenerativeTask(mock(Model.class), stream)
                    .generate("prompt", streamed::add);

            assertThat(result.text()).isEqualTo(" country.");
            assertThat(result.generatedTokens()).isEqualTo(2);
            assertThat(streamed).containsExactly(" country", ".");
        }
    }

    @Test
    void generationStoppedByMaxLengthKeepsEveryToken() throws Exception {
        TokenizerStream stream = mock(TokenizerStream.class);
        when(stream.decode(10)).thenReturn(" country");
        when(stream.decode(13)).thenReturn(".");

        // Every step appends a token; generation ends because the length limit is reached
        try (MockedConstruction<Generator> ignored = mockConstruction(Generator.class, (generator, context) -> {
            when(generator.isDone()).thenReturn(false, false, true);
            when(generator.tokenCount()).thenReturn(4L, 5L, 6L);
            when(generator.getLastTokenInSequence(0)).thenReturn(10, 13);
        })) {
            GenerationResult result = new TestGenerativeTask(mock(Model.class), stream).generate("prompt");

            assertThat(result.text()).isEqualTo(" country.");
            assertThat(result.generatedTokens()).isEqualTo(2);
        }
    }

    /**
     * Minimal concrete subclass for testing the abstract base.
     */
    static class TestGenerativeTask extends AbstractGenerativeTask<String, GenerationResult> {

        private final TokenizerStream stream;
        boolean closeResourcesCalled = false;

        TestGenerativeTask(Model model) {
            this(model, null);
        }

        TestGenerativeTask(Model model, TokenizerStream stream) {
            super(model);
            this.stream = stream;
        }

        @Override
        protected GeneratorParams createParams() {
            return mock(GeneratorParams.class);
        }

        @Override
        protected TokenizerStream createStream() throws GenAIException {
            return stream;
        }

        @Override
        protected void prepareGenerator(String input, Generator generator) {
            // no-op for unit tests
        }

        @Override
        protected GenerationResult parseOutput(String generatedText, String input,
                                               int tokenCount, long durationMillis) {
            return new GenerationResult(generatedText, 0, tokenCount,
                    java.time.Duration.ofMillis(durationMillis));
        }

        @Override
        protected void closeResources() {
            closeResourcesCalled = true;
        }
    }
}
