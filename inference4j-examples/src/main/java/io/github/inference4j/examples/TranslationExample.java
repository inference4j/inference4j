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
import io.github.inference4j.nlp.Language;
import io.github.inference4j.nlp.MarianTranslator;

/**
 * Demonstrates machine translation with MarianMT (fixed pair) and Flan-T5 (flexible).
 *
 * <p>Downloads MarianMT en→fr (~300 MB) and Flan-T5 Small (~300 MB) on first run.
 *
 * <p>Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.TranslationExample
 */
public class TranslationExample {

	public static void main(String[] args) throws Exception {
		String[] sentences = {
				"The weather is beautiful today.",
				"Machine learning is transforming how we build software.",
				"I would like a cup of coffee, please."
		};

		// --- MarianMT: dedicated English→French model ---
		System.out.println("=== MarianMT (en→fr) ===");
		System.out.println();

		try (var translator = MarianTranslator.builder()
				.modelId("inference4j/opus-mt-en-fr")
				.maxNewTokens(100)
				.build()) {

			System.out.println("MarianMT en→fr loaded successfully.");
			System.out.println();

			for (String sentence : sentences) {
				GenerationResult result = translator.translate(sentence, token -> {});

				double seconds = result.duration().toMillis() / 1000.0;
				System.out.printf("  EN: %s%n", sentence);
				System.out.printf("  FR: %s  (%.2fs)%n", result.text(), seconds);
				System.out.println();
			}
		}

		// --- Flan-T5: flexible translation with Language enum ---
		System.out.println("=== Flan-T5 Small (flexible) ===");
		System.out.println();

		try (var translator = FlanT5TextGenerator.flanT5Small()
				.maxNewTokens(100)
				.build()) {

			System.out.println("Flan-T5 Small loaded successfully.");
			System.out.println();

			String text = "The weather is beautiful today.";

			Language[] targets = { Language.PT, Language.ES, Language.DE };
			for (Language target : targets) {
				GenerationResult result = translator.translate(text, Language.EN, target,
						token -> {});

				double seconds = result.duration().toMillis() / 1000.0;
				System.out.printf("  EN → %s: %s  (%.2fs)%n",
						target.displayName(), result.text(), seconds);
			}
		}
	}
}
