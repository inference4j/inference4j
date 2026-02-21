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

package io.github.inference4j.nlp;

import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.ModelSource;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class OnnxTextGeneratorTest {

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThrows(ModelSourceException.class, () ->
                OnnxTextGenerator.builder()
                        .modelSource(badSource)
                        .build());
    }

    @Test
    void builder_directoryMissingModelFile_throws(@TempDir Path dir) throws IOException {
        Files.writeString(dir.resolve("vocab.json"), "{}");
        Files.writeString(dir.resolve("merges.txt"), "");
        Files.writeString(dir.resolve("config.json"), "{}");

        ModelSource source = id -> dir;

        assertThrows(ModelSourceException.class, () ->
                OnnxTextGenerator.builder()
                        .modelSource(source)
                        .build());
    }

    @Test
    void builder_directoryMissingVocab_throws(@TempDir Path dir) throws IOException {
        Files.createFile(dir.resolve("model.onnx"));
        Files.writeString(dir.resolve("merges.txt"), "");
        Files.writeString(dir.resolve("config.json"), "{}");

        ModelSource source = id -> dir;

        assertThrows(ModelSourceException.class, () ->
                OnnxTextGenerator.builder()
                        .modelSource(source)
                        .build());
    }

    @Test
    void builder_directoryMissingMerges_throws(@TempDir Path dir) throws IOException {
        Files.createFile(dir.resolve("model.onnx"));
        Files.writeString(dir.resolve("vocab.json"), "{}");
        Files.writeString(dir.resolve("config.json"), "{}");

        ModelSource source = id -> dir;

        assertThrows(ModelSourceException.class, () ->
                OnnxTextGenerator.builder()
                        .modelSource(source)
                        .build());
    }

    @Test
    void builder_directoryMissingConfig_throws(@TempDir Path dir) throws IOException {
        Files.createFile(dir.resolve("model.onnx"));
        Files.writeString(dir.resolve("vocab.json"), "{}");
        Files.writeString(dir.resolve("merges.txt"), "");

        ModelSource source = id -> dir;

        assertThrows(ModelSourceException.class, () ->
                OnnxTextGenerator.builder()
                        .modelSource(source)
                        .build());
    }

    @Test
    void builder_fluentApi_acceptsAllOptions() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.builder()
                .modelId("custom/model")
                .modelSource(id -> Path.of("/tmp"))
                .sessionOptions(opts -> {})
                .temperature(0.8f)
                .topK(40)
                .topP(0.9f)
                .maxNewTokens(100)
                .eosTokenId(50256)
                .stopSequence("<|endoftext|>")
                .addedToken("<|special|>")
                .tokenizerPattern(OnnxTextGenerator.QWEN2_PATTERN)
                .chatTemplate(msg -> "<|user|>" + msg);

        assertNotNull(builder);
    }

    @Test
    void gpt2_preset_returnsBuilder() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.gpt2();
        assertNotNull(builder);
    }

    @Test
    void smolLM2_preset_returnsBuilder() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.smolLM2();
        assertNotNull(builder);
    }

    @Test
    void qwen2_preset_returnsBuilder() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.qwen2();
        assertNotNull(builder);
    }
}
