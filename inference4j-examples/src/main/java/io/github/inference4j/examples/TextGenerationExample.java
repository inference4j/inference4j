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
package io.github.inference4j.examples;

import io.github.inference4j.genai.GenerationResult;
import io.github.inference4j.genai.GenerativeModel;
import io.github.inference4j.genai.ModelSources;
import io.github.inference4j.nlp.TextGenerator;

/**
 * Compares text generation across models side by side.
 *
 * <p>Runs the same prompts through Phi-3-mini (3.8B) and DeepSeek-R1 (1.5B),
 * showing output quality and generation speed for each.
 *
 * <p>Usage:
 * <pre>
 * ./gradlew :inference4j-examples:run \
 *     -PmainClass=io.github.inference4j.examples.TextGenerationExample
 * </pre>
 */
public class TextGenerationExample {

    private static final String[] PROMPTS = {
            "What is Java in one sentence?",
            "Explain recursion in simple terms.",
    };

    public static void main(String[] args) {
        System.out.println("=== Text Generation — Model Comparison ===\n");

        String[] modelNames = {"Phi-3-mini (3.8B)", "DeepSeek-R1 (1.5B)"};
        GenerativeModel[] models = {ModelSources.phi3Mini(), ModelSources.deepSeekR1_1_5B()};
        GenerationResult[][] results = new GenerationResult[models.length][PROMPTS.length];

        for (int m = 0; m < models.length; m++) {
            System.out.printf("Loading %s...%n", modelNames[m]);
            try (var generator = TextGenerator.builder()
                    .model(models[m])
                    .maxLength(200)
                    .temperature(0.7)
                    .build()) {

                for (int p = 0; p < PROMPTS.length; p++) {
                    results[m][p] = generator.generate(PROMPTS[p]);
                }
            }
        }

        // Print side-by-side results
        for (int p = 0; p < PROMPTS.length; p++) {
            System.out.println("\u2500".repeat(70));
            System.out.printf("Q: %s%n%n", PROMPTS[p]);

            for (int m = 0; m < models.length; m++) {
                GenerationResult r = results[m][p];
                System.out.printf("  [%s]%n", modelNames[m]);
                System.out.printf("  %s%n", r.text().strip());
                System.out.printf("  \u2192 %d tokens in %,d ms (%.1f tok/s)%n%n",
                        r.tokenCount(), r.durationMillis(),
                        r.tokenCount() * 1000.0 / r.durationMillis());
            }
        }
        System.out.println("\u2500".repeat(70));
    }
}
