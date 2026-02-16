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
import io.github.inference4j.nlp.TextGenerator;

/**
 * Demonstrates autoregressive text generation with Phi-3-mini.
 *
 * <p>Usage:
 * <pre>
 * ./gradlew :inference4j-examples:run \
 *     -PmainClass=io.github.inference4j.examples.TextGenerationExample
 * </pre>
 */
public class TextGenerationExample {

    public static void main(String[] args) {
        System.out.println("=== Text Generation with Phi-3-mini ===\n");

        try (var generator = TextGenerator.builder()
                .maxLength(200)
                .temperature(0.7)
                .build()) {

            // Simple generation
            System.out.println("Q: What is Java in one sentence?");
            System.out.print("A: ");
            GenerationResult result = generator.generate(
                    "What is Java in one sentence?",
                    token -> System.out.print(token));
            System.out.printf("%n[%d tokens in %dms]%n%n",
                    result.tokenCount(), result.durationMillis());

            // Second generation reusing the same model
            System.out.println("Q: What is recursion?");
            System.out.print("A: ");
            result = generator.generate(
                    "What is recursion?",
                    token -> System.out.print(token));
            System.out.printf("%n[%d tokens in %dms]%n",
                    result.tokenCount(), result.durationMillis());
        }
    }
}
