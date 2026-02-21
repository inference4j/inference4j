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
package io.github.inference4j.genai.vision;

import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.MultiModalProcessor;
import io.github.inference4j.genai.vision.VisionInput;
import io.github.inference4j.genai.vision.VisionLanguageModel;
import io.github.inference4j.generation.ChatTemplate;
import io.github.inference4j.generation.GenerationResult;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class VisionLanguageModelTest {

	private static final ChatTemplate PHI3V_TEMPLATE =
			msg -> "<|user|>\n<|image_1|>\n" + msg + "\n<|end|>\n<|assistant|>\n";

	@Test
	void parseOutput_stripsWhitespaceAndWrapsInResult() {
		Model model = mock(Model.class);
		MultiModalProcessor processor = mock(MultiModalProcessor.class);

		VisionLanguageModel vlm = new VisionLanguageModel(
				model, processor, PHI3V_TEMPLATE, 1024, 0.0, 0, 0.0);

		GenerationResult result = vlm.parseOutput(
				"  A cat sitting on a mat.  ",
				new VisionInput(Path.of("cat.jpg"), "Describe this image."),
				10, 250);

		assertEquals("A cat sitting on a mat.", result.text());
		assertEquals(10, result.generatedTokens());
		assertEquals(java.time.Duration.ofMillis(250), result.duration());
	}

	@Test
	void closeReleasesProcessorAndModel() {
		Model model = mock(Model.class);
		MultiModalProcessor processor = mock(MultiModalProcessor.class);

		VisionLanguageModel vlm = new VisionLanguageModel(
				model, processor, PHI3V_TEMPLATE, 1024, 0.0, 0, 0.0);
		vlm.close();

		verify(processor).close();
		verify(model).close();
	}

	@Test
	void builderRequiresModelOrModelSource() {
		assertThrows(IllegalStateException.class, () ->
				VisionLanguageModel.builder().build());
	}

	@Test
	void builderRequiresChatTemplateWhenModelIdProvided() {
		assertThrows(IllegalStateException.class, () ->
				VisionLanguageModel.builder()
						.modelId("some/model")
						.build());
	}

	@Test
	void builderAcceptsModelAndProcessorDirectly() {
		Model model = mock(Model.class);
		MultiModalProcessor processor = mock(MultiModalProcessor.class);

		VisionLanguageModel.Builder builder = VisionLanguageModel.builder()
				.chatTemplate(PHI3V_TEMPLATE);
		builder.model = model;
		builder.processor = processor;

		VisionLanguageModel vlm = builder.build();
		assertNotNull(vlm);
	}

	@Test
	void visionInput_recordProperties() {
		Path path = Path.of("photo.jpg");
		VisionInput input = new VisionInput(path, "What is this?");

		assertEquals(path, input.imagePath());
		assertEquals("What is this?", input.prompt());
	}
}
