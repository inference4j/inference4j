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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs all inference4j examples sequentially and prints a pass/fail summary.
 * Exits with non-zero code if any example fails.
 */
public class AllExamplesRunner {

	private interface ExampleMain {
		void run(String[] args) throws Exception;
	}

	public static void main(String[] args) {
		Map<String, ExampleMain> examples = new LinkedHashMap<>();
		examples.put("ImageClassificationExample", ImageClassificationExample::main);
		examples.put("ObjectDetectionExample", ObjectDetectionExample::main);
		examples.put("CraftTextDetectionExample", CraftTextDetectionExample::main);
		examples.put("TextClassificationExample", TextClassificationExample::main);
		examples.put("SemanticSimilarityExample", SemanticSimilarityExample::main);
		examples.put("SemanticSearchExample", SemanticSearchExample::main);
		examples.put("CrossEncoderRerankerExample", CrossEncoderRerankerExample::main);
		examples.put("SpeechToTextExample", SpeechToTextExample::main);
		examples.put("VoiceActivityDetectionExample", VoiceActivityDetectionExample::main);
		examples.put("ModelRouterExample", ModelRouterExample::main);
		examples.put("ModelComparisonExample", ModelComparisonExample::main);
		examples.put("SummarizationExample", SummarizationExample::main);
		examples.put("TranslationExample", TranslationExample::main);
		examples.put("GrammarCorrectionExample", GrammarCorrectionExample::main);

		List<String> failed = new ArrayList<>();

		for (Map.Entry<String, ExampleMain> entry : examples.entrySet()) {
			String name = entry.getKey();
			System.out.println("\n=== " + name + " ===");
			try {
				entry.getValue().run(new String[0]);
				System.out.println("PASSED");
			}
			catch (Exception ex) {
				ex.printStackTrace(System.out);
				System.out.println("FAILED: " + ex.getClass().getSimpleName() + ": " + ex.getMessage());
				failed.add(name);
			}
		}

		int total = examples.size();
		int passed = total - failed.size();
		System.out.println("\n=== Summary: " + passed + "/" + total + " passed ===");
		if (!failed.isEmpty()) {
			System.out.println("  FAILED: " + String.join(", ", failed));
			System.exit(1);
		}
	}

}
