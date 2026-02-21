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

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelMetadataTest {

    @Test
    void property_returnsPresent_forExistingKey() {
        ModelMetadata metadata = new ModelMetadata(
                "pytorch", "graph", "description", 1L,
                Map.of("eos_token_id", "50256", "vocab_size", "50257")
        );

        assertTrue(metadata.property("eos_token_id").isPresent());
        assertEquals("50256", metadata.property("eos_token_id").get());
    }

    @Test
    void property_returnsEmpty_forMissingKey() {
        ModelMetadata metadata = new ModelMetadata(
                "pytorch", "graph", "description", 1L,
                Map.of("key1", "value1")
        );

        assertTrue(metadata.property("nonexistent").isEmpty());
    }

    @Test
    void property_returnsPresent_forAllCustomProperties() {
        Map<String, String> props = Map.of(
                "eos_token_id", "50256",
                "bos_token_id", "50256",
                "model_type", "gpt2"
        );
        ModelMetadata metadata = new ModelMetadata(
                "pytorch", "graph", "", 0L, props
        );

        for (var entry : props.entrySet()) {
            assertEquals(entry.getValue(), metadata.property(entry.getKey()).orElse(null));
        }
    }

    @Test
    void recordFields_areAccessible() {
        ModelMetadata metadata = new ModelMetadata(
                "onnx-exporter", "gpt2-graph", "GPT-2 model", 2L,
                Map.of()
        );

        assertEquals("onnx-exporter", metadata.producerName());
        assertEquals("gpt2-graph", metadata.graphName());
        assertEquals("GPT-2 model", metadata.description());
        assertEquals(2L, metadata.version());
        assertTrue(metadata.customProperties().isEmpty());
    }
}
