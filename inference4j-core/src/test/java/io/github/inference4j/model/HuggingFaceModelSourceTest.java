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

package io.github.inference4j.model;

import io.github.inference4j.exception.ModelSourceException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

class HuggingFaceModelSourceTest {

    @TempDir
    Path tempDir;

    @Test
    void constructor_acceptsCustomCacheDir() {
        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);
        assertThat(source).isNotNull();
    }

    @Test
    void resolve_returnsCachedPathWhenAllRequiredFilesExist() throws IOException {
        Path repoDir = tempDir.resolve("org/model");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("model.onnx"), "fake model");
        Files.writeString(repoDir.resolve("vocab.txt"), "fake vocab");

        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);
        Path resolved = source.resolve("org/model", List.of("model.onnx", "vocab.txt"));

        assertThat(resolved).isEqualTo(repoDir);
    }

    @Test
    void resolve_returnsCachedPathForSingleFile() throws IOException {
        Path repoDir = tempDir.resolve("org/vision");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("vision_model.onnx"), "fake model");

        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);
        Path resolved = source.resolve("org/vision", List.of("vision_model.onnx"));

        assertThat(resolved).isEqualTo(repoDir);
    }

    @Test
    void resolve_singleArg_returnsCachedPathWhenDirectoryExists() throws IOException {
        Path repoDir = tempDir.resolve("org/model");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("model.onnx"), "fake model");

        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);
        Path resolved = source.resolve("org/model");

        assertThat(resolved).isEqualTo(repoDir);
    }

    @Test
    void resolve_singleArg_throwsWhenDirectoryDoesNotExist() {
        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);

        assertThatThrownBy(() -> source.resolve("org/nonexistent"))
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void resolve_triggersDownloadWhenFileMissing() {
        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);

        // Will attempt to download from HuggingFace (no server), but directory gets created
        try {
            source.resolve("org/new-model", List.of("model.onnx"));
        } catch (Exception ignored) {
            // Expected: network call fails
        }

        assertThat(Files.exists(tempDir.resolve("org/new-model"))).isTrue();
    }

    @Test
    void resolve_skipsDownloadWhenAllFilesPresent() throws IOException {
        Path repoDir = tempDir.resolve("org/cached");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("model.onnx"), "fake");
        Files.writeString(repoDir.resolve("labels.txt"), "cat\ndog");

        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);

        // Should return immediately without any HTTP call
        Path resolved = source.resolve("org/cached", List.of("model.onnx", "labels.txt"));
        assertThat(resolved).isEqualTo(repoDir);
    }

    @Test
    void defaultInstance_returnsSameInstance() {
        HuggingFaceModelSource first = HuggingFaceModelSource.defaultInstance();
        HuggingFaceModelSource second = HuggingFaceModelSource.defaultInstance();
        assertThat(second).isSameAs(first);
    }
}
