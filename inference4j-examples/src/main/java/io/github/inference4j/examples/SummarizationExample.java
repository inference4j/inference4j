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
import io.github.inference4j.nlp.BartSummarizer;

/**
 * Demonstrates text summarization with DistilBART CNN 12-6.
 *
 * <p>Downloads the DistilBART CNN ONNX model (~1.2 GB) on first run.
 *
 * <p>Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SummarizationExample
 */
public class SummarizationExample {

	public static void main(String[] args) throws Exception {
		try (var summarizer = BartSummarizer.distilBartCnn()
				.maxNewTokens(100)
				.build()) {

			System.out.println("DistilBART CNN loaded successfully.");
			System.out.println();

			String article = "The Amazon rainforest, often referred to as the lungs of the Earth, "
					+ "produces about 20 percent of the world's oxygen. Spanning across nine countries "
					+ "in South America, it is the largest tropical rainforest in the world, covering "
					+ "approximately 5.5 million square kilometers. The forest is home to an estimated "
					+ "10 percent of all species on Earth, including over 40,000 plant species, 1,300 "
					+ "bird species, and 3,000 types of fish. Deforestation remains a critical threat, "
					+ "with an estimated 17 percent of the forest lost in the last 50 years due to "
					+ "logging, agriculture, and urban expansion.";

			// --- Basic summarization ---
			System.out.println("Input:");
			System.out.println("  " + article);
			System.out.println();

			GenerationResult result = summarizer.summarize(article, token -> {});

			double seconds = result.duration().toMillis() / 1000.0;
			double tokensPerSec = result.generatedTokens() / seconds;

			System.out.println("Summary:");
			System.out.println("  " + result.text());
			System.out.printf("  Tokens: %d  |  Time: %.2fs  |  %.1f tok/s%n",
					result.generatedTokens(), seconds, tokensPerSec);
			System.out.println();

			// --- Streaming summarization ---
			String article2 = "Artificial intelligence has rapidly evolved from a niche academic "
					+ "pursuit into one of the most transformative technologies of the 21st century. "
					+ "Machine learning algorithms now power recommendation systems, autonomous "
					+ "vehicles, medical diagnostics, and language translation. Despite these advances, "
					+ "concerns about bias, privacy, and job displacement continue to fuel debate among "
					+ "policymakers, technologists, and the general public.";

			System.out.println("Streaming summary:");
			System.out.print("  ");
			GenerationResult streamResult = summarizer.summarize(article2,
					token -> System.out.print(token));
			System.out.println();

			seconds = streamResult.duration().toMillis() / 1000.0;
			tokensPerSec = streamResult.generatedTokens() / seconds;
			System.out.printf("  Tokens: %d  |  Time: %.2fs  |  %.1f tok/s%n",
					streamResult.generatedTokens(), seconds, tokensPerSec);
		}
	}
}
