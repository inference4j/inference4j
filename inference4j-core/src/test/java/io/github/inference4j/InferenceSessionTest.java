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

package io.github.inference4j;

import io.github.inference4j.exception.ModelLoadException;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class InferenceSessionTest {

    @Test
    void create_throwsModelLoadExceptionForNonexistentFile() {
        Path badPath = Path.of("/nonexistent/model.onnx");

        ModelLoadException ex = assertThrows(ModelLoadException.class, () ->
                InferenceSession.create(badPath));

        assertTrue(ex.getMessage().contains("/nonexistent/model.onnx"));
    }

    @Test
    void create_throwsModelLoadExceptionForInvalidFile() throws Exception {
        Path tempFile = java.nio.file.Files.createTempFile("not-a-model", ".onnx");
        try {
            java.nio.file.Files.writeString(tempFile, "this is not an onnx model");

            ModelLoadException ex = assertThrows(ModelLoadException.class, () ->
                    InferenceSession.create(tempFile));

            assertTrue(ex.getMessage().contains(tempFile.toString()));
        } finally {
            java.nio.file.Files.deleteIfExists(tempFile);
        }
    }
}
