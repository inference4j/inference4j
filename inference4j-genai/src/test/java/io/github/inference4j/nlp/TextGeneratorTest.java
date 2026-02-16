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

import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.Tokenizer;
import io.github.inference4j.genai.GenerationResult;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class TextGeneratorTest {

    @Test
    void parseOutputWrapsTextInGenerationResult() {
        Model model = mock(Model.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        TextGenerator gen = new TextGenerator(model, tokenizer, 100, 1.0, 0, 0.0);
        GenerationResult result = gen.parseOutput("Hello world", "prompt", 5, 120);

        assertEquals("Hello world", result.text());
        assertEquals(5, result.tokenCount());
        assertEquals(120, result.durationMillis());
    }

    @Test
    void closeReleasesResources() {
        Model model = mock(Model.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        TextGenerator gen = new TextGenerator(model, tokenizer, 100, 1.0, 0, 0.0);
        gen.close();

        verify(tokenizer).close();
        verify(model).close();
    }

    @Test
    void builderRequiresModelSource() {
        TextGenerator.Builder builder = TextGenerator.builder();
        assertNotNull(builder);
        assertThrows(IllegalStateException.class, builder::build);
    }

    @Test
    void buildMessagesJsonEscapesQuotes() {
        Model model = mock(Model.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        TextGenerator gen = new TextGenerator(model, tokenizer, 100, 1.0, 0, 0.0);
        String json = gen.buildMessagesJson("He said \"hello\"");

        assertEquals("[{\"role\": \"user\", \"content\": \"He said \\\"hello\\\"\"}]", json);
    }
}
