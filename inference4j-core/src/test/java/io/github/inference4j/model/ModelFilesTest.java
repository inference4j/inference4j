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

import com.fasterxml.jackson.databind.JsonNode;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.exception.ModelSourceException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ModelFilesTest {

    @TempDir
    Path models;

    private Path modelDir;
    private ModelSource local;

    @BeforeEach
    void setUp() throws IOException {
        modelDir = Files.createDirectory(models.resolve("not-model"));
        try (InputStream model = getClass().getResourceAsStream("/test-bool-not.onnx")) {
            Files.copy(model, modelDir.resolve("model.onnx"));
        }
        local = new LocalModelSource(models);
    }

    @Test
    void loadOpensSessionsThatTheModelKeeps() {
        try (InferenceSession session = ModelFiles.load(local, "not-model", null, List.of("model.onnx"),
                files -> files.session("model.onnx"))) {
            Tensor y = session.run(Map.of("x", Tensor.fromBooleans(new boolean[]{true, false}, new long[]{1, 2})))
                    .get("y");
            assertThat(y.toBooleans()).containsExactly(false, true);
        }
    }

    @Test
    void sessionOptionsAreAppliedToEverySession() {
        AtomicInteger configured = new AtomicInteger();
        try (InferenceSession session = ModelFiles.load(local, "not-model", options -> configured.incrementAndGet(),
                List.of("model.onnx"), files -> files.session("model.onnx"))) {
            assertThat(session.inputNames()).containsExactly("x");
        }
        assertThat(configured).hasValue(1);
    }

    @Test
    void filesAndDirectoryComeFromTheSource() {
        Path directory = ModelFiles.load(local, "not-model", null, List.of(), files -> {
            assertThat(files.modelId()).isEqualTo("not-model");
            assertThat(files.file("model.onnx")).isEqualTo(modelDir.resolve("model.onnx"));
            assertThat(files.lease().directory()).isEqualTo(modelDir);
            return files.directory();
        });
        assertThat(directory).isEqualTo(modelDir);
    }

    @Test
    void missingRequiredFileNamesThePathAndTheModel() {
        assertThatThrownBy(() -> ModelFiles.load(local, "not-model", null, List.of("model.onnx", "config.json"),
                files -> "unreachable"))
                .isInstanceOf(ModelSourceException.class)
                .hasMessage("Model file not found: " + modelDir.resolve("config.json") + " (model not-model)");
    }

    @Test
    void missingDirectoryNamesThePathAndTheModel() {
        Path missing = models.resolve("missing");
        assertThatThrownBy(() -> ModelFiles.load(id -> missing, "acme/missing", null, List.of("model.onnx"),
                files -> "unreachable"))
                .isInstanceOf(ModelSourceException.class)
                .hasMessage("Model directory not found: " + missing + " (model acme/missing)");
    }

    @Test
    void sessionOnAMissingFileFailsLikeARequiredFile() {
        assertThatThrownBy(() -> ModelFiles.load(local, "not-model", null, List.of(),
                files -> files.session("decoder.onnx")))
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining(modelDir.resolve("decoder.onnx").toString());
    }

    @Test
    void uncheckedFilesAreRequestedButNotChecked() {
        List<String> requested = new ArrayList<>();
        ModelSource source = new ModelSource() {
            @Override
            public Path resolve(String modelId) {
                return modelDir;
            }

            @Override
            public ResolvedModel open(String modelId, List<String> files) {
                requested.addAll(files);
                return ResolvedModel.unmanaged(modelDir);
            }
        };

        String result = ModelFiles.load(source, "not-model", null, List.of("model.onnx"), List.of("vocab.json"),
                files -> "built");

        assertThat(result).isEqualTo("built");
        assertThat(requested).containsExactly("model.onnx", "vocab.json");
    }

    @Test
    void failureClosesSessionsAndTrackedResourcesNewestFirst() {
        List<String> closed = new ArrayList<>();
        AtomicReference<InferenceSession> session = new AtomicReference<>();
        IllegalStateException failure = new IllegalStateException("boom");

        assertThatThrownBy(() -> ModelFiles.load(local, "not-model", null, List.of("model.onnx"), files -> {
            session.set(files.session("model.onnx"));
            files.track(() -> closed.add("first"));
            files.track(() -> closed.add("second"));
            throw failure;
        })).isSameAs(failure);

        assertThat(closed).containsExactly("second", "first");
        assertThatThrownBy(() -> session.get().run(Map.of("x",
                Tensor.fromBooleans(new boolean[]{true}, new long[]{1, 1}))))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("closed");
    }

    @Test
    void successLeavesTrackedResourcesOpen() {
        List<String> closed = new ArrayList<>();
        AutoCloseable resource = ModelFiles.load(local, "not-model", null, List.of(),
                files -> files.track(() -> closed.add("resource")));

        assertThat(closed).isEmpty();
        assertThat(resource).isNotNull();
    }

    @Test
    void checkedFailuresBecomeModelLoadExceptions() {
        IOException cause = new IOException("disk on fire");

        assertThatThrownBy(() -> ModelFiles.load(local, "not-model", null, List.of(), files -> {
            throw cause;
        }))
                .isInstanceOf(ModelLoadException.class)
                .hasMessage("Failed to initialize model: disk on fire")
                .hasCause(cause);
    }

    @Test
    void theModelDirectoryIsReleasedWhenLoadingEnds() {
        AtomicInteger released = new AtomicInteger();
        ModelSource source = new ModelSource() {
            @Override
            public Path resolve(String modelId) {
                return modelDir;
            }

            @Override
            public ResolvedModel open(String modelId, List<String> files) {
                return new CountingModel(modelDir, released);
            }
        };

        ModelFiles.load(source, "not-model", null, List.of(), files -> "built");
        assertThat(released).hasValue(1);

        assertThatThrownBy(() -> ModelFiles.load(source, "not-model", null, List.of(), files -> {
            throw new IllegalStateException("boom");
        })).isInstanceOf(IllegalStateException.class);
        assertThat(released).hasValue(2);

        assertThatThrownBy(() -> ModelFiles.load(source, "not-model", null, List.of("missing.onnx"),
                files -> "unreachable")).isInstanceOf(ModelSourceException.class);
        assertThat(released).hasValue(3);
    }

    @Test
    void jsonReadsAFileOfTheModel() throws IOException {
        Files.writeString(modelDir.resolve("config.json"), "{\"eos_token_id\": 2}");

        JsonNode json = ModelFiles.load(local, "not-model", null, List.of(), files -> files.json("config.json"));

        assertThat(json.path("eos_token_id").asInt()).isEqualTo(2);
    }

    @Test
    void malformedJsonIsAModelLoadException() throws IOException {
        Files.writeString(modelDir.resolve("config.json"), "{not json");

        assertThatThrownBy(() -> ModelFiles.load(local, "not-model", null, List.of(),
                files -> files.json("config.json")))
                .isInstanceOf(ModelLoadException.class)
                .hasMessageContaining(modelDir.resolve("config.json").toString());
    }

    @Test
    void missingJsonIsAModelSourceException() {
        assertThatThrownBy(() -> ModelFiles.load(local, "not-model", null, List.of(),
                files -> files.json("config.json")))
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("Model file not found");
    }

    @Test
    void optionalFindsOnlyExistingFiles() {
        List<Optional<Path>> found = ModelFiles.load(local, "not-model", null, List.of(),
                files -> List.of(files.optional("model.onnx"), files.optional("labels.txt")));

        assertThat(found.get(0)).contains(modelDir.resolve("model.onnx"));
        assertThat(found.get(1)).isEmpty();
    }

    @Test
    void closeAllClosesEveryResourceAndRethrowsTheFirstFailure() {
        List<String> closed = new ArrayList<>();
        IllegalStateException first = new IllegalStateException("first");
        IllegalStateException second = new IllegalStateException("second");

        assertThatThrownBy(() -> ModelFiles.closeAll(
                () -> closed.add("a"),
                () -> {
                    closed.add("b");
                    throw first;
                },
                null,
                () -> {
                    closed.add("c");
                    throw second;
                }))
                .isSameAs(first);

        assertThat(closed).containsExactly("a", "b", "c");
        assertThat(first.getSuppressed()).containsExactly(second);
    }

    @Test
    void closeAllWrapsCheckedFailures() {
        Exception cause = new Exception("checked");

        assertThatThrownBy(() -> ModelFiles.closeAll(() -> {
            throw cause;
        }))
                .isInstanceOf(InferenceException.class)
                .hasCause(cause);
    }

    private static final class CountingModel implements ResolvedModel {

        private final Path directory;
        private final AtomicInteger released;

        CountingModel(Path directory, AtomicInteger released) {
            this.directory = directory;
            this.released = released;
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
            return this;
        }

        @Override
        public void close() {
            released.incrementAndGet();
        }
    }
}
