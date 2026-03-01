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

import static org.assertj.core.api.Assertions.*;

class LocalModelSourceTest {

    @TempDir
    Path tempDir;

    @Test
    void resolve_returnsDirectoryPath() throws IOException {
        Path modelDir = tempDir.resolve("my-model");
        Files.createDirectory(modelDir);

        LocalModelSource source = new LocalModelSource(tempDir);
        Path resolved = source.resolve("my-model");

        assertThat(resolved).isEqualTo(modelDir);
        assertThat(Files.isDirectory(resolved)).isTrue();
    }

    @Test
    void resolve_throwsWhenDirectoryNotFound() {
        LocalModelSource source = new LocalModelSource(tempDir);

        assertThatThrownBy(() -> source.resolve("nonexistent"))
                .isInstanceOf(ModelSourceException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains("nonexistent"));
    }

    @Test
    void resolve_throwsWhenPathIsFile() throws IOException {
        Files.writeString(tempDir.resolve("not-a-dir"), "content");

        LocalModelSource source = new LocalModelSource(tempDir);

        assertThatThrownBy(() -> source.resolve("not-a-dir"))
                .isInstanceOf(ModelSourceException.class);
    }
}
