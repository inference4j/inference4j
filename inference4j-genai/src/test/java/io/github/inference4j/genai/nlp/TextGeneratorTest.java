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
package io.github.inference4j.genai.nlp;

import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.Tokenizer;
import io.github.inference4j.genai.nlp.TextGenerator;
import io.github.inference4j.generation.ChatTemplate;
import io.github.inference4j.generation.GenerationResult;
import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Sequences;
import ai.onnxruntime.genai.TokenizerStream;
import io.github.inference4j.generation.ChatMessage;
import ai.onnxruntime.genai.GenAIException;
import org.junit.jupiter.api.Test;
import org.mockito.MockedConstruction;

import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockConstruction;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class TextGeneratorTest {

    private static final ChatTemplate DUMMY_TEMPLATE = msg -> "<|user|>\n" + msg + "<|end|>\n";

    @Test
    void parseOutputWrapsTextInGenerationResult() {
        Model model = mock(Model.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        TextGenerator gen = new TextGenerator(model, tokenizer, DUMMY_TEMPLATE,
                100, 1.0, 0, 0.0);
        GenerationResult result = gen.parseOutput("Hello world", "prompt", 12, 5, 120);

        assertThat(result.text()).isEqualTo("Hello world");
        assertThat(result.promptTokens()).isEqualTo(12);
        assertThat(result.generatedTokens()).isEqualTo(5);
        assertThat(result.duration()).isEqualTo(java.time.Duration.ofMillis(120));
    }

    @Test
    void closeReleasesResources() {
        Model model = mock(Model.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        TextGenerator gen = new TextGenerator(model, tokenizer, DUMMY_TEMPLATE,
                100, 1.0, 0, 0.0);
        gen.close();

        verify(tokenizer).close();
        verify(model).close();
    }

    @Test
    void builderRejectsInvalidOptions() {
        TextGenerator.Builder builder = TextGenerator.builder();
        assertThatThrownBy(() -> builder.intraOpThreads(0)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> builder.maxNewTokens(0)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> builder.repetitionPenalty(0)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> builder.stopSequence("")).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void chatFormatsTheConversationWithTheTemplate() throws Exception {
        Tokenizer tokenizer = tokenizer();
        TextGenerator gen = new TextGenerator(mock(Model.class), tokenizer,
                io.github.inference4j.genai.ModelSources.phi4Mini().chatTemplate(), 100, 0, 0, 0);

        try (MockedConstruction<GeneratorParams> params = mockConstruction(GeneratorParams.class);
             MockedConstruction<Generator> generators = oneTokenGenerator()) {
            GenerationResult result = gen.chat(List.of(ChatMessage.system("Be brief."), ChatMessage.user("Hi")));

            assertThat(result.text()).isEqualTo("Hello");
            assertThat(result.promptTokens()).isEqualTo(4);
        }
        verify(tokenizer).encode("<|system|>Be brief.<|end|><|user|>Hi<|end|><|assistant|>");
    }

    @Test
    void chatNeedsATemplateThatFormatsConversations() {
        TextGenerator gen = new TextGenerator(mock(Model.class), mock(Tokenizer.class), DUMMY_TEMPLATE,
                100, 0, 0, 0);

        try (MockedConstruction<GeneratorParams> params = mockConstruction(GeneratorParams.class);
             MockedConstruction<Generator> generators = oneTokenGenerator()) {
            assertThatThrownBy(() -> gen.chat(List.of(ChatMessage.user("Hi"))))
                    .isInstanceOf(UnsupportedOperationException.class)
                    .hasMessageContaining("format(List<ChatMessage>)");
        }
    }

    @Test
    void chatRejectsAnEmptyConversation() {
        TextGenerator gen = new TextGenerator(mock(Model.class), mock(Tokenizer.class), DUMMY_TEMPLATE,
                100, 0, 0, 0);

        assertThatThrownBy(() -> gen.chat(List.of())).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void exposesItsLimits() {
        TextGenerator gen = new TextGenerator(mock(Model.class), mock(Tokenizer.class), DUMMY_TEMPLATE,
                512, 0, 0, 0, 64, 0, Set.of());

        assertThat(gen.maxLength()).isEqualTo(512);
        assertThat(gen.maxNewTokens()).isEqualTo(64);
    }

    private static Tokenizer tokenizer() throws GenAIException {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString())).thenReturn(mock(Sequences.class));
        TokenizerStream stream = mock(TokenizerStream.class);
        when(stream.decode(10)).thenReturn("Hello");
        when(tokenizer.createStream()).thenReturn(stream);
        return tokenizer;
    }

    /** Generators produce one token, 10, after a 4-token prompt. */
    private static MockedConstruction<Generator> oneTokenGenerator() {
        return mockConstruction(Generator.class, (generator, context) -> {
            when(generator.isDone()).thenReturn(false, true);
            when(generator.tokenCount()).thenReturn(4L, 5L);
            when(generator.getLastTokenInSequence(0)).thenReturn(10);
        });
    }

    @Test
    void builderRequiresModel() {
        TextGenerator.Builder builder = TextGenerator.builder();
        assertThat(builder).isNotNull();
        assertThatThrownBy(builder::build).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void builderRequiresChatTemplateWhenModelSourceProvided() {
        TextGenerator.Builder builder = TextGenerator.builder()
                .modelSource(modelId -> null);
        assertThatThrownBy(builder::build).isInstanceOf(IllegalStateException.class);
    }
}
