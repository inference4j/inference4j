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

import io.github.inference4j.genai.ModelSources;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.generation.GenerativeModel;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Qwen3-0.6B, SmolLM3-3B and Phi-4-mini through their presets. Downloads about 7.3 GB on first run.
 */
class NewerTextModelsModelTest {

    static Stream<Arguments> presets() {
        return Stream.of(
                Arguments.of("qwen3-0.6b", (Supplier<GenerativeModel>) ModelSources::qwen3_0_6B),
                Arguments.of("smollm3-3b", (Supplier<GenerativeModel>) ModelSources::smolLM3_3B),
                Arguments.of("phi-4-mini", (Supplier<GenerativeModel>) ModelSources::phi4Mini));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("presets")
    void answersDirectlyAndStopsAtTheEndOfTheTurn(String name, Supplier<GenerativeModel> preset) throws Exception {
        try (var gen = TextGenerator.builder().model(preset.get()).maxLength(256).build()) {
            List<String> streamed = new ArrayList<>();
            GenerationResult result = gen.generate("What is the capital of France? Answer in one sentence.", streamed::add);

            assertThat(result.text()).contains("Paris");
            // Non-thinking presets answer directly; and the turn ends instead of running on into a new one
            assertThat(result.text()).doesNotContain("<think>", "</think>", "<|im_start|>", "<|user|>");
            assertThat(result.generatedTokens()).isLessThan(40);
            assertThat(String.join("", streamed)).isEqualTo(result.text());
        }
    }
}
