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
import io.github.inference4j.nlp.FlanT5TextGenerator;

/**
 * Demonstrates grammar correction with Flan-T5 Small.
 *
 * <p>Downloads the Flan-T5 Small ONNX model (~300 MB) on first run.
 *
 * <p>Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.GrammarCorrectionExample
 */
public class GrammarCorrectionExample {

	public static void main(String[] args) throws Exception {
		String[] sentences = {
				"She don't likes swimming.",
				"Me and him went to the store yesterday.",
				"The informations is very useful for we.",
				"I have went to the park last week.",
				"Their going to the movies tonight.",
		};

		try (var corrector = FlanT5TextGenerator.flanT5Small()
				.maxNewTokens(100)
				.build()) {

			System.out.println("Flan-T5 Small loaded successfully.");
			System.out.println();

			for (String sentence : sentences) {
				GenerationResult result = corrector.correct(sentence, token -> {});

				double seconds = result.duration().toMillis() / 1000.0;
				System.out.printf("  Input:     %s%n", sentence);
				System.out.printf("  Corrected: %s  (%.2fs)%n", result.text(), seconds);
				System.out.println();
			}
		}
	}
}
