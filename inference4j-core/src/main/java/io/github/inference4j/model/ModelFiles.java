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
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.session.SessionConfigurer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * The files of a model while a builder loads it: the opened model directory, the sessions and other resources
 * created from it so far, and the rules every builder follows to get there.
 *
 * <p><strong>Internal support for inference4j's own model wrappers.</strong> It is public only so that wrappers
 * in different packages can share it; it may change or disappear in any release without notice.
 *
 * <p>A builder calls {@link #load(ModelSource, String, SessionConfigurer, List, Assembler) load}, which:
 * <ol>
 *   <li>opens the model through its {@link ModelSource} ({@link HuggingFaceModelSource} when none is given), which
 *       holds the directory in the cache until loading ends;</li>
 *   <li>checks that the directory and every required file exist, failing with a {@link ModelSourceException} that
 *       names the path and the model;</li>
 *   <li>passes a {@code ModelFiles} to the {@link Assembler}, which reads files and opens sessions through it and
 *       returns the model.</li>
 * </ol>
 *
 * <p>When the assembler returns, the sessions and {@linkplain #track tracked} resources belong to the model, and
 * only the hold on the directory is released (each session keeps its own). When anything fails, every session and
 * tracked resource is closed, newest first, and the failure propagates: runtime exceptions as they are, checked
 * exceptions wrapped in a {@link ModelLoadException}.
 *
 * <p>A {@code ModelFiles} lives only for one {@code load} call; a model never keeps it. Not thread-safe.
 */
public final class ModelFiles implements AutoCloseable {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /**
     * Builds a model from its files.
     *
     * @param <T> the model type
     */
    @FunctionalInterface
    public interface Assembler<T> {

        /**
         * Builds the model. Sessions and resources opened through {@code files} are closed if this throws.
         *
         * @param files the model's files
         * @return the model, which now owns the sessions and tracked resources
         * @throws Exception if the model cannot be built
         */
        T assemble(ModelFiles files) throws Exception;
    }

    private final ResolvedModel model;
    private final String modelId;
    private final SessionConfigurer sessionConfigurer;
    private final List<AutoCloseable> resources = new ArrayList<>();
    private boolean assembled;
    private boolean closed;

    private ModelFiles(ResolvedModel model, String modelId, SessionConfigurer sessionConfigurer) {
        this.model = model;
        this.modelId = modelId;
        this.sessionConfigurer = sessionConfigurer;
    }

    /**
     * Opens a model, checks its required files and builds it.
     *
     * @param source            where the model comes from; {@code null} for {@link HuggingFaceModelSource}
     * @param modelId           the model identifier passed to the source
     * @param sessionConfigurer customizes every session opened with {@link #session(String)}; {@code null} for the
     *                          defaults
     * @param required          files requested from the source, each of which must exist
     * @param assembler         builds the model
     * @param <T>               the model type
     * @return the model
     * @throws ModelSourceException if the model, its directory or a required file cannot be found
     * @throws ModelLoadException   if the assembler fails with a checked exception
     */
    public static <T> T load(ModelSource source, String modelId, SessionConfigurer sessionConfigurer,
                             List<String> required, Assembler<T> assembler) {
        return load(source, modelId, sessionConfigurer, required, List.of(), assembler);
    }

    /**
     * Like {@link #load(ModelSource, String, SessionConfigurer, List, Assembler)}, also requesting
     * {@code unchecked} files from the source without checking that they exist: the assembler, or whatever reads
     * them, deals with their absence. A downloading source may still fail when it cannot supply them.
     *
     * @param source            where the model comes from; {@code null} for {@link HuggingFaceModelSource}
     * @param modelId           the model identifier passed to the source
     * @param sessionConfigurer customizes every session; {@code null} for the defaults
     * @param required          files requested from the source, each of which must exist
     * @param unchecked         further files requested from the source, not checked
     * @param assembler         builds the model
     * @param <T>               the model type
     * @return the model
     */
    public static <T> T load(ModelSource source, String modelId, SessionConfigurer sessionConfigurer,
                             List<String> required, List<String> unchecked, Assembler<T> assembler) {
        ModelSource resolved = source != null ? source : HuggingFaceModelSource.defaultInstance();
        List<String> requested = new ArrayList<>(required);
        requested.addAll(unchecked);
        return load(resolved.open(modelId, requested), modelId, sessionConfigurer, required, assembler);
    }

    /**
     * Like {@link #load(ModelSource, String, SessionConfigurer, List, Assembler)}, for a model the caller has
     * already opened, for builders that open it in their own way. This call takes over {@code model} and closes it.
     *
     * @param model             the opened model
     * @param modelId           the model identifier, for messages
     * @param sessionConfigurer customizes every session; {@code null} for the defaults
     * @param required          files that must exist
     * @param assembler         builds the model
     * @param <T>               the model type
     * @return the model
     */
    public static <T> T load(ResolvedModel model, String modelId, SessionConfigurer sessionConfigurer,
                             List<String> required, Assembler<T> assembler) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(assembler, "assembler");
        return assemble(new ModelFiles(model, modelId, sessionConfigurer), required, assembler);
    }

    private static <T> T assemble(ModelFiles opened, List<String> required, Assembler<T> assembler) {
        try (ModelFiles files = opened) {
            files.check(required);
            T result = assembler.assemble(files);
            files.assembled = true;
            return result;
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new ModelLoadException("Failed to initialize model: " + e.getMessage(), e);
        }
    }

    /** The model identifier, as given to the source. */
    public String modelId() {
        return modelId;
    }

    /** The model directory. */
    public Path directory() {
        return model.directory();
    }

    /** A file in the model directory, by its path relative to it; it may not exist. */
    public Path file(String name) {
        return model.file(name);
    }

    /** A file in the model directory, if it exists. */
    public Optional<Path> optional(String name) {
        Path file = file(name);
        return Files.exists(file) ? Optional.of(file) : Optional.empty();
    }

    /**
     * Reads a JSON file of the model.
     *
     * @throws ModelSourceException if the file does not exist
     * @throws ModelLoadException   if the file cannot be read or is not valid JSON
     */
    public JsonNode json(String name) {
        Path file = require(name);
        try {
            return MAPPER.readTree(file.toFile());
        } catch (IOException e) {
            throw new ModelLoadException("Failed to read " + file + ": " + e.getMessage(), e);
        }
    }

    /**
     * Opens an ONNX Runtime session on a model file, with the builder's session options. The session is closed if
     * loading fails; otherwise it belongs to the model.
     *
     * @throws ModelSourceException if the file does not exist
     */
    public InferenceSession session(String name) {
        require(name);
        InferenceSession session = sessionConfigurer != null
                ? InferenceSession.create(model, name, sessionConfigurer)
                : InferenceSession.create(model, name);
        return track(session);
    }

    /**
     * Registers a resource to close if loading fails; on success it belongs to the model. Resources are closed
     * newest first, so a resource made from another is freed before it.
     *
     * @return {@code resource}
     */
    public <R extends AutoCloseable> R track(R resource) {
        resources.add(Objects.requireNonNull(resource, "resource"));
        return resource;
    }

    /**
     * The opened model directory. It is released when loading ends; {@linkplain ResolvedModel#retain() retain} it
     * to keep the files beyond that. Do not close it.
     */
    public ResolvedModel lease() {
        return model;
    }

    /**
     * Ends loading: closes the sessions and tracked resources unless the model was built, then releases the
     * directory. Called by {@code load}; closing twice does nothing.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        List<AutoCloseable> toClose = new ArrayList<>();
        if (!assembled) {
            for (int i = resources.size() - 1; i >= 0; i--) {
                toClose.add(resources.get(i));
            }
        }
        toClose.add(model);
        closeAll(toClose.toArray(new AutoCloseable[0]));
    }

    /**
     * Closes every resource in the given order, even when one fails, skipping {@code null}s. The first failure is
     * rethrown with the others {@linkplain Throwable#addSuppressed suppressed}: a runtime exception as it is, a
     * checked one wrapped in an {@link InferenceException}.
     *
     * @param resources the resources to close
     */
    public static void closeAll(AutoCloseable... resources) {
        Exception failure = null;
        for (AutoCloseable resource : resources) {
            if (resource == null) {
                continue;
            }
            try {
                resource.close();
            } catch (Exception e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure != null) {
            throw new InferenceException("Failed to close a resource: " + failure.getMessage(), failure);
        }
    }

    private void check(List<String> required) {
        Path directory = directory();
        if (!Files.isDirectory(directory)) {
            throw new ModelSourceException("Model directory not found: " + directory + forModel());
        }
        for (String name : required) {
            require(name);
        }
    }

    private Path require(String name) {
        Path file = file(name);
        if (!Files.exists(file)) {
            throw new ModelSourceException("Model file not found: " + file + forModel());
        }
        return file;
    }

    private String forModel() {
        return modelId == null || modelId.isEmpty() ? "" : " (model " + modelId + ")";
    }
}
