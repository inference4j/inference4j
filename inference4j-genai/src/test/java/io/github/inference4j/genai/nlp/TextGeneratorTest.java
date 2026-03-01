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
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class TextGeneratorTest {

    private static final ChatTemplate DUMMY_TEMPLATE = msg -> "<|user|>\n" + msg + "<|end|>\n";

    @Test
    void parseOutputWrapsTextInGenerationResult() {
        Model model = mock(Model.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        TextGenerator gen = new TextGenerator(model, tokenizer, DUMMY_TEMPLATE,
                100, 1.0, 0, 0.0);
        GenerationResult result = gen.parseOutput("Hello world", "prompt", 5, 120);

        assertThat(result.text()).isEqualTo("Hello world");
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
