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

import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.Tokenizer;
import org.junit.jupiter.api.Test;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class AbstractGenerativeTaskTest {

    @Test
    void closeReleasesModelAndTokenizer() throws Exception {
        Model model = mock(Model.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        var task = new TestGenerativeTask(model, tokenizer, 100, 1.0, 0, 0.0);
        task.close();

        verify(tokenizer).close();
        verify(model).close();
    }

    /**
     * Minimal concrete subclass for testing the abstract base.
     */
    static class TestGenerativeTask extends AbstractGenerativeTask<String, GenerationResult> {

        TestGenerativeTask(Model model, Tokenizer tokenizer,
                           int maxLength, double temperature, int topK, double topP) {
            super(model, tokenizer, maxLength, temperature, topK, topP);
        }

        @Override
        protected void prepareGenerator(String input, Generator generator) {
            // no-op for unit tests
        }

        @Override
        protected GenerationResult parseOutput(String generatedText, String input,
                                               int tokenCount, long durationMillis) {
            return new GenerationResult(generatedText, tokenCount, durationMillis);
        }
    }
}
