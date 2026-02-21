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
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.TokenizerStream;
import io.github.inference4j.generation.GenerationResult;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

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

        assertTrue(task.closeResourcesCalled,
                "closeResources() should be called during close()");
        verify(model).close();
    }

    /**
     * Minimal concrete subclass for testing the abstract base.
     */
    static class TestGenerativeTask extends AbstractGenerativeTask<String, GenerationResult> {

        boolean closeResourcesCalled = false;

        TestGenerativeTask(Model model) {
            super(model);
        }

        @Override
        protected TokenizerStream createStream() throws GenAIException {
            return null;
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
