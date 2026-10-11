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

package io.github.inference4j.autoconfigure;

import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;

import io.github.inference4j.generation.GenerativeModel;
import io.github.inference4j.genai.ModelSources;
import io.github.inference4j.genai.audio.WhisperSpeechModel;
import io.github.inference4j.genai.audio.WhisperTask;
import io.github.inference4j.genai.nlp.TextGenerator;
import io.github.inference4j.genai.vision.VisionLanguageModel;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Lazy;

/**
 * Auto-configuration for the generative models of {@code inference4j-genai}, active when that module is
 * on the classpath.
 *
 * <p>Runs before {@link Inference4jAudioAutoConfiguration}: {@link WhisperSpeechModel} is a
 * {@link io.github.inference4j.audio.SpeechRecognizer}, so enabling Whisper makes it the application's speech
 * recognizer and the Wav2Vec2 bean backs off.
 */
@AutoConfiguration(before = Inference4jAudioAutoConfiguration.class)
@ConditionalOnClass(TextGenerator.class)
@EnableConfigurationProperties(Inference4jProperties.class)
public class Inference4jGenaiAutoConfiguration {

	private static final Map<String, Supplier<GenerativeModel>> TEXT_MODELS = Map.of(
			"phi-3-mini", ModelSources::phi3Mini,
			"deepseek-r1-1.5b", ModelSources::deepSeekR1_1_5B);

	private static final Map<String, Supplier<GenerativeModel>> VISION_MODELS = Map.of(
			"phi-3.5-vision", ModelSources::phi3Vision);

	@Bean
	@Lazy
	@ConditionalOnMissingBean(TextGenerator.class)
	@ConditionalOnProperty(prefix = "inference4j.genai.text-generator", name = "enabled", havingValue = "true")
	public TextGenerator textGenerator(Inference4jProperties properties) {
		Inference4jProperties.GenerationProperties config = properties.getGenai().getTextGenerator();
		TextGenerator.Builder builder = TextGenerator.builder()
			.model(preset(TEXT_MODELS, config.getModel(), "inference4j.genai.text-generator.model"));
		if (config.getMaxLength() != null) {
			builder.maxLength(config.getMaxLength());
		}
		if (config.getTemperature() != null) {
			builder.temperature(config.getTemperature());
		}
		if (config.getTopK() != null) {
			builder.topK(config.getTopK());
		}
		if (config.getTopP() != null) {
			builder.topP(config.getTopP());
		}
		return builder.build();
	}

	@Bean
	@Lazy
	@ConditionalOnMissingBean(VisionLanguageModel.class)
	@ConditionalOnProperty(prefix = "inference4j.genai.vision-language", name = "enabled", havingValue = "true")
	public VisionLanguageModel visionLanguageModel(Inference4jProperties properties) {
		Inference4jProperties.GenerationProperties config = properties.getGenai().getVisionLanguage();
		VisionLanguageModel.Builder builder = VisionLanguageModel.builder()
			.model(preset(VISION_MODELS, config.getModel(), "inference4j.genai.vision-language.model"));
		if (config.getMaxLength() != null) {
			builder.maxLength(config.getMaxLength());
		}
		if (config.getTemperature() != null) {
			builder.temperature(config.getTemperature());
		}
		if (config.getTopK() != null) {
			builder.topK(config.getTopK());
		}
		if (config.getTopP() != null) {
			builder.topP(config.getTopP());
		}
		return builder.build();
	}

	@Bean
	@Lazy
	@ConditionalOnMissingBean(WhisperSpeechModel.class)
	@ConditionalOnProperty(prefix = "inference4j.genai.whisper", name = "enabled", havingValue = "true")
	public WhisperSpeechModel whisperSpeechModel(Inference4jProperties properties) {
		Inference4jProperties.WhisperProperties config = properties.getGenai().getWhisper();
		WhisperSpeechModel.Builder builder = WhisperSpeechModel.builder()
			.modelId(config.getModelId())
			.language(config.getLanguage())
			.task(whisperTask(config.getTask()));
		if (config.getMaxLength() != null) {
			builder.maxLength(config.getMaxLength());
		}
		return builder.build();
	}

	static GenerativeModel preset(Map<String, Supplier<GenerativeModel>> presets, String name, String property) {
		Supplier<GenerativeModel> preset = name != null ? presets.get(name.toLowerCase(Locale.ROOT)) : null;
		if (preset == null) {
			throw new IllegalArgumentException("Unknown " + property + " '" + name + "'. Supported: "
					+ presets.keySet().stream().sorted().toList()
					+ ". For other models, define your own bean.");
		}
		return preset.get();
	}

	static WhisperTask whisperTask(String task) {
		try {
			return WhisperTask.valueOf(task.toUpperCase(Locale.ROOT));
		}
		catch (IllegalArgumentException | NullPointerException ex) {
			throw new IllegalArgumentException(
					"Unknown inference4j.genai.whisper.task '" + task + "'. Supported: [transcribe, translate]", ex);
		}
	}

}
