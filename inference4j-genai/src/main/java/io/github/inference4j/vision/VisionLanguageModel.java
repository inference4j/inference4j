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
package io.github.inference4j.vision;

import ai.onnxruntime.genai.GenAIException;
import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Images;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.MultiModalProcessor;
import ai.onnxruntime.genai.NamedTensors;
import ai.onnxruntime.genai.TokenizerStream;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.genai.AbstractGenerativeTask;
import io.github.inference4j.genai.ChatTemplate;
import io.github.inference4j.genai.GenerationResult;
import io.github.inference4j.genai.GenerativeModel;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;

import java.nio.file.Path;
import java.util.function.Consumer;

/**
 * Vision-language model backed by onnxruntime-genai.
 *
 * <p>Wraps multimodal models (Phi-3 Vision, Phi-3.5 Vision) that accept an
 * image and a text prompt and generate text output. Image preprocessing,
 * vision encoding, autoregressive decoding, KV cache, and sampling are all
 * handled natively by onnxruntime-genai's C++ layer.
 *
 * <h2>Describe an image</h2>
 * <pre>{@code
 * try (var vision = VisionLanguageModel.builder()
 *         .model(ModelSources.phi3Vision())
 *         .build()) {
 *     GenerationResult result = vision.describe(Path.of("photo.jpg"));
 *     System.out.println(result.text());
 * }
 * }</pre>
 *
 * <h2>Ask a question about an image</h2>
 * <pre>{@code
 * try (var vision = VisionLanguageModel.builder()
 *         .model(ModelSources.phi3Vision())
 *         .build()) {
 *     vision.ask(Path.of("chart.png"), "What trend does this chart show?",
 *             token -> System.out.print(token));
 * }
 * }</pre>
 *
 * @see VisionInput
 * @see io.github.inference4j.genai.ModelSources#phi3Vision()
 */
public class VisionLanguageModel extends AbstractGenerativeTask<VisionInput, GenerationResult> {

	private final MultiModalProcessor processor;
	private final ChatTemplate chatTemplate;
	private final int maxLength;
	private final double temperature;
	private final int topK;
	private final double topP;

	VisionLanguageModel(Model model, MultiModalProcessor processor,
						ChatTemplate chatTemplate,
						int maxLength, double temperature, int topK, double topP) {
		super(model);
		this.processor = processor;
		this.chatTemplate = chatTemplate;
		this.maxLength = maxLength;
		this.temperature = temperature;
		this.topK = topK;
		this.topP = topP;
	}

	public static Builder builder() {
		return new Builder();
	}

	/**
	 * Describes an image using a default prompt.
	 *
	 * @param imagePath path to an image file (PNG, JPEG, etc.)
	 * @return the generated description
	 */
	public GenerationResult describe(Path imagePath) {
		return generate(new VisionInput(imagePath, "Describe this image."));
	}

	/**
	 * Describes an image using a default prompt, streaming tokens as they are generated.
	 *
	 * @param imagePath     path to an image file (PNG, JPEG, etc.)
	 * @param tokenListener receives each decoded text fragment as it is generated
	 * @return the complete generation result
	 */
	public GenerationResult describe(Path imagePath, Consumer<String> tokenListener) {
		return generate(new VisionInput(imagePath, "Describe this image."), tokenListener);
	}

	/**
	 * Asks a question about an image.
	 *
	 * @param imagePath path to an image file (PNG, JPEG, etc.)
	 * @param question  the question to ask about the image
	 * @return the generated answer
	 */
	public GenerationResult ask(Path imagePath, String question) {
		return generate(new VisionInput(imagePath, question));
	}

	/**
	 * Asks a question about an image, streaming tokens as they are generated.
	 *
	 * @param imagePath     path to an image file (PNG, JPEG, etc.)
	 * @param question      the question to ask about the image
	 * @param tokenListener receives each decoded text fragment as it is generated
	 * @return the complete generation result
	 */
	public GenerationResult ask(Path imagePath, String question,
								Consumer<String> tokenListener) {
		return generate(new VisionInput(imagePath, question), tokenListener);
	}

	@Override
	protected GeneratorParams createParams() throws GenAIException {
		GeneratorParams params = super.createParams();
		params.setSearchOption("max_length", maxLength);
		if (temperature > 0) {
			params.setSearchOption("temperature", temperature);
		}
		if (topK > 0) {
			params.setSearchOption("top_k", topK);
		}
		if (topP > 0) {
			params.setSearchOption("top_p", topP);
		}
		return params;
	}

	@Override
	protected void prepareGenerator(VisionInput input, Generator generator) {
		try {
			try (Images images = new Images(input.imagePath().toString())) {
				String prompt = chatTemplate.format(input.prompt());
				try (NamedTensors inputs = processor.processImages(prompt, images)) {
					generator.setInputs(inputs);
				}
			}
		} catch (GenAIException e) {
			throw new InferenceException(
					"Failed to prepare vision generator: " + e.getMessage(), e);
		}
	}

	@Override
	protected TokenizerStream createStream() throws GenAIException {
		return processor.createStream();
	}

	@Override
	protected GenerationResult parseOutput(String generatedText, VisionInput input,
										   int tokenCount, long durationMillis) {
		return new GenerationResult(generatedText.strip(), tokenCount, durationMillis);
	}

	@Override
	protected void closeResources() {
		processor.close();
	}

	/**
	 * Builder for {@link VisionLanguageModel}.
	 *
	 * <p>Minimal usage with a preconfigured model:
	 * <pre>{@code
	 * VisionLanguageModel vision = VisionLanguageModel.builder()
	 *         .model(ModelSources.phi3Vision())
	 *         .build();
	 * }</pre>
	 *
	 * <p>Custom model:
	 * <pre>{@code
	 * VisionLanguageModel vision = VisionLanguageModel.builder()
	 *         .modelId("my-org/my-vision-model")
	 *         .chatTemplate(msg -> "<|user|>\n<|image_1|>\n" + msg + "\n<|end|>\n<|assistant|>\n")
	 *         .build();
	 * }</pre>
	 */
	public static class Builder {

		private ModelSource modelSource;
		private ChatTemplate chatTemplate;
		private String modelId;
		private int maxLength = 4096;
		private double temperature = 0.0;
		private int topK = 0;
		private double topP = 0.0;

		// Package-private for testing
		Model model;
		MultiModalProcessor processor;

		public Builder model(GenerativeModel generativeModel) {
			this.modelSource = generativeModel.modelSource();
			this.chatTemplate = generativeModel.chatTemplate();
			return this;
		}

		public Builder modelId(String modelId) {
			this.modelId = modelId;
			return this;
		}

		public Builder modelSource(ModelSource modelSource) {
			this.modelSource = modelSource;
			return this;
		}

		public Builder chatTemplate(ChatTemplate chatTemplate) {
			this.chatTemplate = chatTemplate;
			return this;
		}

		public Builder maxLength(int maxLength) {
			this.maxLength = maxLength;
			return this;
		}

		public Builder temperature(double temperature) {
			this.temperature = temperature;
			return this;
		}

		public Builder topK(int topK) {
			this.topK = topK;
			return this;
		}

		public Builder topP(double topP) {
			this.topP = topP;
			return this;
		}

		public VisionLanguageModel build() {
			if (model == null) {
				if (modelSource == null && modelId == null) {
					throw new IllegalStateException(
							"model is required — use model(ModelSources.phi3Vision()) "
									+ "or provide modelId + chatTemplate");
				}
				if (chatTemplate == null) {
					throw new IllegalStateException(
							"chatTemplate is required — use model(ModelSources.phi3Vision()) "
									+ "or provide a chatTemplate alongside modelSource/modelId");
				}
				ModelSource source = modelSource != null
						? modelSource : HuggingFaceModelSource.defaultInstance();
				String id = modelId != null ? modelId : "model";
				Path modelDir = source.resolve(id);
				Model m = null;
				try {
					m = new Model(modelDir.toString());
					model = m;
					processor = new MultiModalProcessor(m);
				} catch (GenAIException e) {
					if (m != null) {
						m.close();
					}
					throw new ModelSourceException(
							"Failed to load vision model: " + e.getMessage(), e);
				}
			}
			return new VisionLanguageModel(model, processor, chatTemplate,
					maxLength, temperature, topK, topP);
		}
	}
}
