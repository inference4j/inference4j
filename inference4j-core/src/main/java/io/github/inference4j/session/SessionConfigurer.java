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

package io.github.inference4j.session;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.inference4j.InferenceSession;

/**
 * Configures ONNX Runtime session options before session creation.
 *
 * <p>This functional interface allows users to customize session options
 * (e.g., GPU execution providers, thread counts) without exposing
 * {@link InferenceSession} internals. The configurer runs after default
 * options are applied (thread counts, optimization level), so it can
 * override defaults or add execution providers.
 *
 * <p>Unlike {@code Consumer<OrtSession.SessionOptions>}, this interface
 * declares {@link OrtException}, allowing clean lambdas for methods like
 * {@code addCUDA()} and {@code addCoreML()} that throw checked exceptions.
 *
 * <p>Example:
 * <pre>{@code
 * ResNetClassifier.builder()
 *     .sessionOptions(opts -> opts.addCUDA(0))
 *     .build();
 * }</pre>
 *
 * @see InferenceSession#create(java.nio.file.Path, SessionConfigurer)
 */
@FunctionalInterface
public interface SessionConfigurer {

    /**
     * Configures the given session options.
     *
     * @param options the session options to configure
     * @throws OrtException if configuration fails (e.g., CUDA not available)
     */
    void configure(OrtSession.SessionOptions options) throws OrtException;
}
