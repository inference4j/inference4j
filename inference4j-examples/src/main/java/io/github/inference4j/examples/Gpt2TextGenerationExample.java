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

import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.nlp.OnnxTextGenerator;

/**
 * Demonstrates text generation with GPT-2 (124M parameters).
 *
 * <p>Downloads the GPT-2 ONNX model (~500 MB) on first run.
 *
 * <p>Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.Gpt2TextGenerationExample
 */
public class Gpt2TextGenerationExample {

    public static void main(String[] args) throws Exception {
        try (var gen = OnnxTextGenerator.gpt2()
                .maxNewTokens(50)
                .build()) {

            System.out.println("GPT-2 loaded successfully.");
            System.out.println();

            // --- Basic generation ---
            String[] prompts = {
                    "Once upon a time",
                    "The meaning of life is"
            };

            for (String prompt : prompts) {
                GenerationResult result = gen.generate(prompt);

                double seconds = result.duration().toMillis() / 1000.0;
                double tokensPerSec = result.generatedTokens() / seconds;

                System.out.printf("Prompt: \"%s\"%n", prompt);
                System.out.printf("  Generated: %s%n", result.text());
                System.out.printf("  Tokens: %d  |  Time: %.2fs  |  %.1f tok/s%n",
                        result.generatedTokens(), seconds, tokensPerSec);
                System.out.println();
            }

            // --- Streaming generation ---
            System.out.println("Streaming: \"The quick brown fox\"");
            System.out.print("  ");
            GenerationResult streamResult = gen.generate("The quick brown fox",
                    token -> System.out.print(token));
            System.out.println();

            double seconds = streamResult.duration().toMillis() / 1000.0;
            double tokensPerSec = streamResult.generatedTokens() / seconds;
            System.out.printf("  Tokens: %d  |  Time: %.2fs  |  %.1f tok/s%n",
                    streamResult.generatedTokens(), seconds, tokensPerSec);
        }
    }
}
