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

package io.github.inference4j;

import java.nio.file.Path;

/**
 * Strategy for resolving a model identifier to a local directory.
 *
 * <p>Implementations handle model discovery and download from various sources
 * (local filesystem, HuggingFace Hub, S3, etc.). The resolved directory is
 * expected to contain at least a {@code model.onnx} file.
 *
 * <p>Example implementation for a local cache:
 * <pre>{@code
 * ModelSource local = modelId -> Path.of("/models", modelId);
 * ResNet model = ResNet.fromPretrained("resnet50", local);
 * }</pre>
 *
 * @see InferenceSession#create(Path)
 */
@FunctionalInterface
public interface ModelSource {

    /**
     * Resolves a model identifier to a local directory path.
     *
     * @param modelId the model identifier (e.g., {@code "resnet50"}, {@code "bert-base-uncased"})
     * @return path to a local directory containing the model files
     * @throws io.github.inference4j.exception.ModelSourceException if the model cannot be resolved
     */
    Path resolve(String modelId);
}
