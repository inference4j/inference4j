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

import java.util.Map;
import java.util.Optional;

/**
 * Metadata embedded in an ONNX model file.
 *
 * <p>Wraps the metadata returned by ONNX Runtime's session, providing
 * convenient access to standard fields and custom key-value properties.
 *
 * @param producerName     the name of the tool that produced the model (e.g., "pytorch")
 * @param graphName        the name of the model graph
 * @param description      free-text description embedded in the model
 * @param version          the model version number
 * @param customProperties custom key-value metadata embedded in the model
 */
public record ModelMetadata(
        String producerName,
        String graphName,
        String description,
        long version,
        Map<String, String> customProperties
) {

    /**
     * Looks up a custom metadata property by key.
     *
     * @param key the property key
     * @return the value if present, empty otherwise
     */
    public Optional<String> property(String key) {
        return Optional.ofNullable(customProperties.get(key));
    }
}
