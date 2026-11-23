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

import io.github.inference4j.session.SessionConfigurer;

import java.util.List;

/**
 * Base class of model builders: the options every builder shares ({@link #modelId(String) modelId},
 * {@link #modelSource(ModelSource) modelSource} and {@link #sessionOptions(SessionConfigurer) sessionOptions}) and
 * loading through {@link ModelFiles}.
 *
 * <p><strong>Internal support for inference4j's own model wrappers.</strong> It is public only so that wrappers
 * in different packages can share it; it may change or disappear in any release without notice.
 *
 * @param <T> the model type the builder builds
 * @param <B> the concrete builder type, returned by the setters for chaining
 */
public abstract class AbstractModelBuilder<T, B extends AbstractModelBuilder<T, B>> {

    /** The model identifier set with {@link #modelId(String)}, or {@code null}. */
    protected String modelId;
    /** The model source set with {@link #modelSource(ModelSource)}, or {@code null} for HuggingFace. */
    protected ModelSource modelSource;
    /** The session options set with {@link #sessionOptions(SessionConfigurer)}, or {@code null}. */
    protected SessionConfigurer sessionConfigurer;

    private final String defaultModelId;

    /** A builder with no default model: one must be set. */
    protected AbstractModelBuilder() {
        this(null);
    }

    /**
     * A builder with a default model.
     *
     * @param defaultModelId the model used when {@link #modelId(String)} is not set
     */
    protected AbstractModelBuilder(String defaultModelId) {
        this.defaultModelId = defaultModelId;
    }

    /** The model to load: a HuggingFace repository by default, or an id the {@link ModelSource} understands. */
    public B modelId(String modelId) {
        this.modelId = modelId;
        return self();
    }

    /** Where to find the model. Defaults to HuggingFace ({@link HuggingFaceModelSource}). */
    public B modelSource(ModelSource modelSource) {
        this.modelSource = modelSource;
        return self();
    }

    /** Customizes the ONNX Runtime options of the model's sessions. */
    public B sessionOptions(SessionConfigurer sessionConfigurer) {
        this.sessionConfigurer = sessionConfigurer;
        return self();
    }

    /** Loads the model. */
    public abstract T build();

    /** This builder, as its concrete type. */
    @SuppressWarnings("unchecked")
    protected final B self() {
        return (B) this;
    }

    /** The model set with {@link #modelId(String)}, or the builder's default; {@code null} when neither is set. */
    protected String resolvedModelId() {
        return modelId != null ? modelId : defaultModelId;
    }

    /** The model source set with {@link #modelSource(ModelSource)}, or {@link HuggingFaceModelSource}. */
    protected ModelSource source() {
        return modelSource != null ? modelSource : HuggingFaceModelSource.defaultInstance();
    }

    /**
     * Loads the {@linkplain #resolvedModelId() model} with {@link ModelFiles}.
     *
     * @param required  files that must exist
     * @param assembler builds the model
     * @return the model
     */
    protected T load(List<String> required, ModelFiles.Assembler<T> assembler) {
        return load(resolvedModelId(), required, assembler);
    }

    /**
     * Loads a model with {@link ModelFiles}, for builders that derive the identifier themselves.
     *
     * @param modelId   the model identifier
     * @param required  files that must exist
     * @param assembler builds the model
     * @return the model
     */
    protected T load(String modelId, List<String> required, ModelFiles.Assembler<T> assembler) {
        return load(modelId, required, List.of(), assembler);
    }

    /**
     * Loads a model with {@link ModelFiles}, also requesting {@code unchecked} files from the source without
     * checking that they exist.
     *
     * @param modelId   the model identifier
     * @param required  files that must exist
     * @param unchecked further files to request from the source
     * @param assembler builds the model
     * @return the model
     * @see ModelFiles#load(ModelSource, String, SessionConfigurer, List, List, ModelFiles.Assembler)
     */
    protected T load(String modelId, List<String> required, List<String> unchecked,
                     ModelFiles.Assembler<T> assembler) {
        return ModelFiles.load(source(), modelId, sessionConfigurer, required, unchecked, assembler);
    }
}
