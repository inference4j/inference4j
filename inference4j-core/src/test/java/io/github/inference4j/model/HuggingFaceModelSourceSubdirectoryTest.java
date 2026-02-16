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

import static org.junit.jupiter.api.Assertions.assertEquals;

class HuggingFaceModelSourceSubdirectoryTest {

    @TempDir
    Path cacheDir;

    @Test
    void resolveWithSubdirectoryReturnsCachedPathWhenPresent() throws IOException {
        // Simulate a cached genai model directory
        Path repoDir = cacheDir.resolve("microsoft/Phi-3-mini-4k-instruct-onnx");
        Path subDir = repoDir.resolve("cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4");
        Files.createDirectories(subDir);
        Files.writeString(subDir.resolve("genai_config.json"), "{}");
        Files.writeString(subDir.resolve("model.onnx"), "fake");

        HuggingFaceModelSource source = new HuggingFaceModelSource(cacheDir);
        Path resolved = source.resolve(
                "microsoft/Phi-3-mini-4k-instruct-onnx",
                "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4");

        assertEquals(subDir, resolved);
    }

    @Test
    void defaultResolveWithSubdirectoryDelegatesToResolve() {
        // ModelSource.resolve(id, subdir) default delegates to resolve(id)
        // and appends the subdirectory
        ModelSource local = id -> cacheDir.resolve(id);
        Path result = local.resolve("my-model", "sub/dir");
        assertEquals(cacheDir.resolve("my-model").resolve("sub/dir"), result);
    }
}
