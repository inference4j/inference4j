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
    void constructorAcceptsCustomCacheDir() {
        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);
        assertThat(source).isNotNull();
    }

    @Test
    void resolveAdoptsTheFlatCacheWhenAllRequiredFilesExist() throws IOException {
        Path repoDir = tempDir.resolve("org/model");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("model.onnx"), "fake model");
        Files.writeString(repoDir.resolve("vocab.txt"), "fake vocab");

        HuggingFaceModelSource source = offlineSource(tempDir);
        Path resolved = source.resolve("org/model", List.of("model.onnx", "vocab.txt"));

        assertThat(resolved).isEqualTo(adopted("org/model"));
        assertThat(Files.readString(resolved.resolve("vocab.txt"))).isEqualTo("fake vocab");
    }

    @Test
    void resolveAdoptsTheFlatCacheForSingleFile() throws IOException {
        Path repoDir = tempDir.resolve("org/vision");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("vision_model.onnx"), "fake model");

        HuggingFaceModelSource source = offlineSource(tempDir);
        Path resolved = source.resolve("org/vision", List.of("vision_model.onnx"));

        assertThat(resolved).isEqualTo(adopted("org/vision"));
        assertThat(resolved.resolve("vision_model.onnx")).exists();
    }

    @Test
    void resolveWithoutFilesAdoptsTheFlatCacheDirectory() throws IOException {
        Path repoDir = tempDir.resolve("org/model");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("model.onnx"), "fake model");

        HuggingFaceModelSource source = offlineSource(tempDir);
        Path resolved = source.resolve("org/model");

        assertThat(resolved).isEqualTo(adopted("org/model"));
        assertThat(resolved.resolve("model.onnx")).exists();
    }

    @Test
    void resolveWithoutFilesThrowsWhenNothingIsCached() {
        HuggingFaceModelSource source = offlineSource(tempDir);

        assertThatThrownBy(() -> source.resolve("org/nonexistent"))
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void resolveSkipsDownloadWhenAllFilesPresent() throws IOException {
        Path repoDir = tempDir.resolve("org/cached");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("model.onnx"), "fake");
        Files.writeString(repoDir.resolve("labels.txt"), "cat\ndog");

        HuggingFaceModelSource source = offlineSource(tempDir);

        // Should return immediately without any HTTP call
        Path resolved = source.resolve("org/cached", List.of("model.onnx", "labels.txt"));
        assertThat(resolved).isEqualTo(adopted("org/cached"));
    }

    @Test
    void resolveWithoutFilesReturnsTheCurrentSnapshot() throws IOException {
        Path repoDir = tempDir.resolve("org/model");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("model.onnx"), "fake model");
        HuggingFaceModelSource source = offlineSource(tempDir);
        Path first = source.resolve("org/model", List.of("model.onnx"));

        try (ResolvedModel model = source.open("org/model")) {
            assertThat(model.directory()).isEqualTo(first);
            assertThat(model.revision()).contains(new ModelRevision("org/model", ModelRevision.UNKNOWN));
        }
    }

    @Test
    void defaultInstanceReturnsSameInstance() {
        HuggingFaceModelSource first = HuggingFaceModelSource.defaultInstance();
        HuggingFaceModelSource second = HuggingFaceModelSource.defaultInstance();
        assertThat(second).isSameAs(first);
    }

    /** Where files of the flat layout are adopted to when offline. */
    private Path adopted(String repoId) {
        return tempDir.resolve("hub").resolve("models--" + repoId.replace("/", "--")).resolve("snapshots")
                .resolve(ModelRevision.UNKNOWN);
    }

    /** A source that never contacts the Hub, so these cache tests make no network calls. */
    private static HuggingFaceModelSource offlineSource(Path cacheDir) {
        return HuggingFaceModelSource.builder().cacheDir(cacheDir).offline(true).build();
    }
}
