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

package io.github.inference4j.audio;

import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.LocalModelSource;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

class MoonshineRecognizerTest {

    @TempDir
    Path models;

    @Test
    void buildFailsWhenTheModelDirectoryIsMissing() {
        assertThatThrownBy(() -> MoonshineRecognizer.builder()
                .modelSource(new LocalModelSource(models))
                .modelId("moonshine")
                .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void buildNamesTheMissingFile() throws IOException {
        Path dir = Files.createDirectory(models.resolve("moonshine"));
        Files.writeString(dir.resolve("streaming_config.json"), "{}");

        assertThatThrownBy(() -> MoonshineRecognizer.builder()
                .modelSource(new LocalModelSource(models))
                .modelId("moonshine")
                .build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("frontend.model.ort");
    }
}
