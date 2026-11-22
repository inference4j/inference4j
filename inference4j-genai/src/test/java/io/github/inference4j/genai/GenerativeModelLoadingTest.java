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
package io.github.inference4j.genai;

import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.MultiModalProcessor;
import ai.onnxruntime.genai.Tokenizer;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.genai.audio.WhisperSpeechModel;
import io.github.inference4j.genai.nlp.TextGenerator;
import io.github.inference4j.genai.vision.VisionLanguageModel;
import io.github.inference4j.generation.ChatTemplate;
import io.github.inference4j.model.LocalModelSource;
import io.github.inference4j.model.ModelRevision;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.model.ResolvedModel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.MockedConstruction;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mockConstruction;
import static org.mockito.Mockito.verify;

class GenerativeModelLoadingTest {

    private static final ChatTemplate TEMPLATE = message -> message;

    @TempDir
    Path models;

    private Path modelDir;
    private CountingSource source;

    @BeforeEach
    void setUp() throws IOException {
        modelDir = Files.createDirectory(models.resolve("tiny"));
        source = new CountingSource(modelDir);
    }

    @Test
    void textGeneratorReportsAMissingGenaiConfigAndReleasesTheDirectory() {
        assertThatThrownBy(() -> TextGenerator.builder().modelSource(source).chatTemplate(TEMPLATE).build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessage("Model file not found: " + modelDir.resolve("genai_config.json"));

        assertThat(source.requested).containsExactly("model");
        assertThat(source.held).hasValue(0);
    }

    @Test
    void visionLanguageModelReportsAMissingGenaiConfigWithTheModelId() {
        assertThatThrownBy(() -> VisionLanguageModel.builder()
                .modelSource(source).modelId("my-org/vision").chatTemplate(TEMPLATE).build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessage("Model file not found: " + modelDir.resolve("genai_config.json")
                        + " (model my-org/vision)");

        assertThat(source.requested).containsExactly("my-org/vision");
        assertThat(source.held).hasValue(0);
    }

    @Test
    void whisperRequestsItsFilesAndReportsAMissingGenaiConfig() {
        assertThatThrownBy(() -> WhisperSpeechModel.builder().modelSource(source).modelId("tiny").build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessage("Model file not found: " + modelDir.resolve("genai_config.json") + " (model tiny)");

        assertThat(source.requested).containsExactly("tiny");
        assertThat(source.requestedFiles).contains("genai_config.json", "encoder.onnx", "decoder.onnx");
        assertThat(source.held).hasValue(0);
    }

    @Test
    void aMissingLocalModelDirectoryIsReported() {
        Path missing = models.resolve("missing");

        assertThatThrownBy(() -> WhisperSpeechModel.builder()
                .modelSource(new LocalModelSource(models)).modelId("missing").build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageStartingWith("Model directory not found: " + missing);
    }

    @Test
    void anUnreadableGenaiConfigIsAModelLoadException() throws IOException {
        Files.writeString(modelDir.resolve("genai_config.json"), "{not json");

        assertThatThrownBy(() -> TextGenerator.builder()
                .modelSource(source).chatTemplate(TEMPLATE).intraOpThreads(2).build())
                .isInstanceOf(ModelLoadException.class)
                .hasMessageContaining(modelDir.resolve("genai_config.json").toString());

        assertThat(source.held).hasValue(0);
    }

    @Test
    void theTaskHoldsTheDirectoryUntilItIsClosed() throws IOException {
        Files.writeString(modelDir.resolve("genai_config.json"), "{}");

        try (MockedConstruction<Model> loaded = mockConstruction(Model.class);
             MockedConstruction<Tokenizer> tokenizers = mockConstruction(Tokenizer.class)) {
            TextGenerator generator = TextGenerator.builder().modelSource(source).chatTemplate(TEMPLATE).build();
            assertThat(source.held).hasValue(1);

            generator.close();

            assertThat(source.held).hasValue(0);
            verify(loaded.constructed().get(0)).close();
            verify(tokenizers.constructed().get(0)).close();
        }
    }

    @Test
    void aFailureAfterTheModelLoadsClosesItAndReleasesTheDirectory() throws IOException {
        Files.writeString(modelDir.resolve("genai_config.json"), "{}");
        IllegalStateException failure = new IllegalStateException("processor failed");

        try (MockedConstruction<Model> loaded = mockConstruction(Model.class);
             MockedConstruction<MultiModalProcessor> processors = mockConstruction(MultiModalProcessor.class,
                     (processor, context) -> {
                         throw failure;
                     })) {
            assertThatThrownBy(() -> WhisperSpeechModel.builder().modelSource(source).modelId("tiny").build())
                    .isInstanceOf(RuntimeException.class);

            assertThat(loaded.constructed()).hasSize(1);
            verify(loaded.constructed().get(0)).close();
            assertThat(source.held).hasValue(0);
        }
    }

    /** Opens {@code directory} whatever is asked for, counting the handles still held on it. */
    private static final class CountingSource implements ModelSource {

        private final Path directory;
        private final AtomicInteger held = new AtomicInteger();
        private final List<String> requested = new ArrayList<>();
        private final List<String> requestedFiles = new ArrayList<>();

        CountingSource(Path directory) {
            this.directory = directory;
        }

        @Override
        public Path resolve(String modelId) {
            return directory;
        }

        @Override
        public ResolvedModel open(String modelId) {
            requested.add(modelId);
            return new Handle();
        }

        @Override
        public ResolvedModel open(String modelId, List<String> files) {
            requestedFiles.addAll(files);
            return open(modelId);
        }

        private final class Handle implements ResolvedModel {

            private boolean closed;

            Handle() {
                held.incrementAndGet();
            }

            @Override
            public Path directory() {
                return directory;
            }

            @Override
            public Optional<ModelRevision> revision() {
                return Optional.empty();
            }

            @Override
            public ResolvedModel retain() {
                return new Handle();
            }

            @Override
            public synchronized void close() {
                if (!closed) {
                    closed = true;
                    held.decrementAndGet();
                }
            }
        }
    }
}
