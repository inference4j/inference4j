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
import io.github.inference4j.session.SessionOptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.*;

class InferenceSessionTest {

    @TempDir
    Path tempDir;

    @Test
    void create_throwsModelLoadExceptionForNonexistentFile() {
        Path badPath = Path.of("/nonexistent/model.onnx");

        assertThatThrownBy(() -> InferenceSession.create(badPath))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains("/nonexistent/model.onnx"));
    }

    @Test
    void create_throwsModelLoadExceptionForInvalidFile() throws IOException {
        Path tempFile = tempDir.resolve("not-a-model.onnx");
        Files.writeString(tempFile, "this is not an onnx model");

        assertThatThrownBy(() -> InferenceSession.create(tempFile))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains(tempFile.toString()));
    }

    @Test
    void create_withSessionOptions_throwsForNonexistentFile() {
        Path badPath = Path.of("/nonexistent/model.onnx");
        SessionOptions options = SessionOptions.defaults();

        assertThatThrownBy(() -> InferenceSession.create(badPath, options))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains("/nonexistent/model.onnx"));
    }

    @Test
    void create_withSessionOptions_throwsForInvalidFile() throws IOException {
        Path tempFile = tempDir.resolve("bad-model.onnx");
        Files.writeString(tempFile, "garbage data");
        SessionOptions options = SessionOptions.defaults();

        assertThatThrownBy(() -> InferenceSession.create(tempFile, options))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains(tempFile.toString()));
    }

    @Test
    void create_withConfigurer_throwsForNonexistentFile() {
        Path badPath = Path.of("/nonexistent/model.onnx");

        assertThatThrownBy(() -> InferenceSession.create(badPath, opts -> { }))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains("/nonexistent/model.onnx"));
    }

    @Test
    void create_withConfigurer_throwsForInvalidFile() throws IOException {
        Path tempFile = tempDir.resolve("invalid.onnx");
        Files.writeString(tempFile, "not valid onnx");

        assertThatThrownBy(() -> InferenceSession.create(tempFile, opts -> { }))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains(tempFile.toString()));
    }

    @Test
    void create_modelLoadExceptionHasCause() {
        Path badPath = Path.of("/nonexistent/model.onnx");

        assertThatThrownBy(() -> InferenceSession.create(badPath))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getCause()).isNotNull());
    }

    @Test
    void create_withCustomSessionOptions_throwsForInvalidFile() throws IOException {
        Path tempFile = tempDir.resolve("custom-opts.onnx");
        Files.writeString(tempFile, "not onnx");

        SessionOptions options = SessionOptions.builder()
                .intraOpNumThreads(2)
                .interOpNumThreads(1)
                .build();

        assertThatThrownBy(() -> InferenceSession.create(tempFile, options))
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains(tempFile.toString()));
    }
}
