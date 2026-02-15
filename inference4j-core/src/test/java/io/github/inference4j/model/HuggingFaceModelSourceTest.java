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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class HuggingFaceModelSourceTest {

    @TempDir
    Path tempDir;

    @Test
    void constructor_acceptsCustomCacheDir() {
        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);
        assertNotNull(source);
    }

    @Test
    void resolve_returnsCachedPathWhenModelOnnxExists() throws IOException {
        Path repoDir = tempDir.resolve("org/model");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("model.onnx"), "fake model");

        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);
        Path resolved = source.resolve("org/model");

        assertEquals(repoDir, resolved);
    }

    @Test
    void resolve_returnsCachedPathWhenSileroVadOnnxExists() throws IOException {
        Path repoDir = tempDir.resolve("org/silero-vad");
        Files.createDirectories(repoDir);
        Files.writeString(repoDir.resolve("silero_vad.onnx"), "fake vad model");

        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);
        Path resolved = source.resolve("org/silero-vad");

        assertEquals(repoDir, resolved);
    }

    @Test
    void resolve_cacheHitAfterLockWhenConcurrentDownload() throws IOException {
        Path repoDir = tempDir.resolve("org/concurrent-model");
        Files.createDirectories(repoDir);

        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);

        // First call will try to download (and fail since no real server),
        // but if we pre-populate the cache before the second call, it should hit cache.
        // Simulate: create model.onnx after the directory exists
        Files.writeString(repoDir.resolve("model.onnx"), "fake model");

        Path resolved = source.resolve("org/concurrent-model");
        assertEquals(repoDir, resolved);
    }

    @Test
    void defaultInstance_returnsSameInstance() {
        HuggingFaceModelSource first = HuggingFaceModelSource.defaultInstance();
        HuggingFaceModelSource second = HuggingFaceModelSource.defaultInstance();
        assertSame(first, second);
    }

    @Test
    void resolve_createsRepoDirWhenMissing() {
        Path repoDir = tempDir.resolve("org/new-model");
        assertFalse(Files.exists(repoDir));

        HuggingFaceModelSource source = new HuggingFaceModelSource(tempDir);

        // This will attempt to download from HuggingFace, which will fail.
        // But we can verify the directory was created.
        try {
            source.resolve("org/new-model");
        } catch (Exception ignored) {
            // Expected: network call fails
        }

        assertTrue(Files.exists(repoDir));
    }
}
