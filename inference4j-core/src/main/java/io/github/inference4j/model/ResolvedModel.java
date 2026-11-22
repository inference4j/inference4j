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

import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/**
 * A model directory resolved by a {@link ModelSource}, held open while it is in use.
 *
 * <p>For a {@link HuggingFaceModelSource}, the directory is a snapshot of the repo at one commit, and an open
 * handle keeps that snapshot on disk: updates write new snapshots next to it, and
 * {@link ModelCache#evict eviction} and pruning skip it until every handle on it is closed. Other sources return
 * {@linkplain #unmanaged(Path) unmanaged} handles, whose {@link #close()} does nothing.
 *
 * <p>Model builders hold a handle while they load, and each {@link io.github.inference4j.InferenceSession}
 * created from it {@linkplain #retain() retains} its own, released when the session closes:
 * <pre>{@code
 * try (ResolvedModel model = source.open("inference4j/silero-vad", List.of("model.onnx"))) {
 *     session = InferenceSession.create(model, "model.onnx");
 * }
 * // the session keeps the snapshot until session.close()
 * }</pre>
 *
 * <p>Thread-safe. Closing a handle twice releases it once.
 */
public interface ResolvedModel extends AutoCloseable {

    /** The directory holding the model files. */
    Path directory();

    /** A file in the {@linkplain #directory() directory}, by its path relative to it. */
    default Path file(String name) {
        return directory().resolve(name);
    }

    /** The repo revision the files belong to; empty for unmanaged directories. */
    Optional<ModelRevision> revision();

    /**
     * Returns a new handle on the same directory that keeps it in use until that handle is closed, independently
     * of this one.
     *
     * @throws IllegalStateException if this handle was closed
     */
    ResolvedModel retain();

    /** Releases this handle. Closing twice does nothing. */
    @Override
    void close();

    /** A handle on a directory no cache manages: it has no revision, and closing it does nothing. */
    static ResolvedModel unmanaged(Path directory) {
        Objects.requireNonNull(directory, "directory");
        return new ResolvedModel() {
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
            }

            @Override
            public String toString() {
                return "ResolvedModel[" + directory + "]";
            }
        };
    }
}
