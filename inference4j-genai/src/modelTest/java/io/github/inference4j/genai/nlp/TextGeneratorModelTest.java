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
package io.github.inference4j.genai.nlp;

import io.github.inference4j.genai.nlp.TextGenerator;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.genai.ModelSources;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

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

            assertThat(result).isNotNull();
            assertThat(result.text()).isNotNull();
            assertThat(result.text().isBlank()).as("Generated text should not be blank").isFalse();
            assertThat(result.generatedTokens()).as("Should generate at least one token").isGreaterThan(0);
            assertThat(result.duration()).as("Duration should not be null").isNotNull();
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

            assertThat(streamedTokens).as("Should stream at least one token").isNotEmpty();
            String streamed = String.join("", streamedTokens);
            assertThat(result.text().contains(streamed) || streamed.contains(result.text()))
                    .as("Streamed tokens should match final result").isTrue();
        }
    }
}
