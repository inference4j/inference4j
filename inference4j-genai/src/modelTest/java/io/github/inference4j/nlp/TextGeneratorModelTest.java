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

import io.github.inference4j.genai.GenerationResult;
import io.github.inference4j.genai.ModelSources;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration test that runs real text generation with Phi-3-mini.
 *
 * <p>Requires the model to be downloaded locally. The model will be
 * auto-downloaded from HuggingFace on first run (requires network, ~2GB).
 */
class TextGeneratorModelTest {

    @Test
    void generateProducesNonEmptyOutput() {
        try (var generator = TextGenerator.builder()
                .model(ModelSources.phi3Mini())
                .maxLength(100)
                .temperature(0.7)
                .build()) {

            GenerationResult result = generator.generate("What is 2 + 2?");

            assertNotNull(result);
            assertNotNull(result.text());
            assertFalse(result.text().isBlank(), "Generated text should not be blank");
            assertTrue(result.tokenCount() > 0, "Should generate at least one token");
            assertTrue(result.durationMillis() >= 0, "Duration should be non-negative");
        }
    }

    @Test
    void generateWithStreamingCollectsAllTokens() {
        try (var generator = TextGenerator.builder()
                .model(ModelSources.phi3Mini())
                .maxLength(50)
                .build()) {

            List<String> streamedTokens = new ArrayList<>();
            GenerationResult result = generator.generate(
                    "Say hello.",
                    streamedTokens::add);

            assertFalse(streamedTokens.isEmpty(), "Should stream at least one token");
            String streamed = String.join("", streamedTokens);
            assertTrue(result.text().contains(streamed) || streamed.contains(result.text()),
                    "Streamed tokens should match final result");
        }
    }
}
