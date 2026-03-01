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

import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.tokenizer.DecodingBpeTokenizer;
import io.github.inference4j.tokenizer.SentencePieceBpeTokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.*;

class OnnxTextGeneratorTest {

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                OnnxTextGenerator.builder()
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_directoryMissingModelFile_throws(@TempDir Path dir) throws IOException {
        Files.writeString(dir.resolve("vocab.json"), "{}");
        Files.writeString(dir.resolve("merges.txt"), "");
        Files.writeString(dir.resolve("config.json"), "{}");

        ModelSource source = id -> dir;

        assertThatThrownBy(() ->
                OnnxTextGenerator.builder()
                        .modelSource(source)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_directoryMissingVocab_throws(@TempDir Path dir) throws IOException {
        Files.createFile(dir.resolve("model.onnx"));
        Files.writeString(dir.resolve("merges.txt"), "");
        Files.writeString(dir.resolve("config.json"), "{}");

        ModelSource source = id -> dir;

        assertThatThrownBy(() ->
                OnnxTextGenerator.builder()
                        .modelSource(source)
                        .build())
                .isInstanceOf(RuntimeException.class);
    }

    @Test
    void builder_directoryMissingMerges_throws(@TempDir Path dir) throws IOException {
        Files.createFile(dir.resolve("model.onnx"));
        Files.writeString(dir.resolve("vocab.json"), "{}");
        Files.writeString(dir.resolve("config.json"), "{}");

        ModelSource source = id -> dir;

        assertThatThrownBy(() ->
                OnnxTextGenerator.builder()
                        .modelSource(source)
                        .build())
                .isInstanceOf(RuntimeException.class);
    }

    @Test
    void builder_directoryMissingConfig_throws(@TempDir Path dir) throws IOException {
        Files.createFile(dir.resolve("model.onnx"));
        Files.writeString(dir.resolve("vocab.json"), "{}");
        Files.writeString(dir.resolve("merges.txt"), "");

        ModelSource source = id -> dir;

        assertThatThrownBy(() ->
                OnnxTextGenerator.builder()
                        .modelSource(source)
                        .build())
                .isInstanceOf(ModelSourceException.class);
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
                .tokenizerProvider(DecodingBpeTokenizer.provider(OnnxTextGenerator.QWEN2_PATTERN))
                .chatTemplate(msg -> "<|user|>" + msg);

        assertThat(builder).isNotNull();
    }

    @Test
    void builder_fluentApi_acceptsTokenizerProvider() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.builder()
                .modelId("custom/model")
                .tokenizerProvider(SentencePieceBpeTokenizer.provider());

        assertThat(builder).isNotNull();
    }

    @Test
    void gpt2_preset_returnsBuilder() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.gpt2();
        assertThat(builder).isNotNull();
    }

    @Test
    void smolLM2_preset_returnsBuilder() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.smolLM2();
        assertThat(builder).isNotNull();
    }

    @Test
    void qwen2_preset_returnsBuilder() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.qwen2();
        assertThat(builder).isNotNull();
    }

    @Test
    void gemma2_preset_returnsBuilder() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.gemma2();
        assertThat(builder).isNotNull();
    }

    @Test
    void tinyLlama_preset_returnsBuilder() {
        OnnxTextGenerator.Builder builder = OnnxTextGenerator.tinyLlama();
        assertThat(builder).isNotNull();
    }

    @Test
    void builder_noModelIdOrSource_throws() {
        assertThatThrownBy(() -> OnnxTextGenerator.builder().build())
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> {
                    assertThat(ex.getMessage()).contains("modelId");
                    assertThat(ex.getMessage()).contains("modelSource");
                });
    }

    @Test
    void gemma2_preset_requiresModelSource() {
        assertThatThrownBy(() -> OnnxTextGenerator.gemma2().build())
                .isInstanceOf(ModelLoadException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains("modelSource"));
    }
}
