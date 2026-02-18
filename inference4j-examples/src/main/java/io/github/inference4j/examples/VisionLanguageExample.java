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
import io.github.inference4j.genai.ModelSources;
import io.github.inference4j.vision.VisionInput;
import io.github.inference4j.vision.VisionLanguageModel;

import java.nio.file.Path;

/**
 * Demonstrates image description and visual Q&A with Phi-3.5 Vision.
 *
 * <p>Requires a sample image at {@code assets/images/sample.jpg} — see
 * inference4j-examples/README.md for setup.
 *
 * <p>Usage:
 * <pre>
 * ./gradlew :inference4j-examples:run \
 *     -PmainClass=io.github.inference4j.examples.VisionLanguageExample
 * </pre>
 */
public class VisionLanguageExample {

    public static void main(String[] args) {
        Path imagePath = Path.of("assets/images/sample.jpg");

        System.out.println("=== Vision Language Model — Phi-3.5 Vision ===");
        System.out.printf("Image: %s%n%n", imagePath);

        try (var vision = VisionLanguageModel.builder()
                .model(ModelSources.phi3Vision())
                .maxLength(4096)
                .build()) {

            System.out.println("Phi-3.5 Vision loaded successfully.\n");

            // Describe the image
            System.out.print("Description: ");
            GenerationResult description = vision.generate(
                    new VisionInput(imagePath, "Describe this image."),
                    token -> System.out.print(token));
            System.out.printf("%n→ %d tokens in %,d ms (%.1f tok/s)%n%n",
                    description.tokenCount(), description.durationMillis(),
                    description.tokenCount() * 1000.0 / description.durationMillis());

            // Ask a question about the image
            String question = "What colors are prominent in this image?";
            System.out.printf("Q: %s%nA: ", question);
            GenerationResult answer = vision.generate(
                    new VisionInput(imagePath, question),
                    token -> System.out.print(token));
            System.out.printf("%n→ %d tokens in %,d ms (%.1f tok/s)%n",
                    answer.tokenCount(), answer.durationMillis(),
                    answer.tokenCount() * 1000.0 / answer.durationMillis());
        }
    }
}
