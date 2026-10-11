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

import io.github.inference4j.audio.SpeechRecognizer;
import io.github.inference4j.genai.audio.WhisperSpeechModel;
import io.github.inference4j.genai.nlp.TextGenerator;
import io.github.inference4j.genai.vision.VisionLanguageModel;
import org.junit.jupiter.api.Test;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.FilteredClassLoader;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;

class Inference4jGenaiAutoConfigurationTest {

	private final ApplicationContextRunner runner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(Inference4jGenaiAutoConfiguration.class));

	@Test
	void noBeansByDefault() {
		runner.run(ctx -> {
			assertThat(ctx).doesNotHaveBean(TextGenerator.class);
			assertThat(ctx).doesNotHaveBean(VisionLanguageModel.class);
			assertThat(ctx).doesNotHaveBean(WhisperSpeechModel.class);
		});
	}

	@Test
	void backsOffWhenInference4jGenaiIsNotOnTheClasspath() {
		runner.withClassLoader(new FilteredClassLoader(TextGenerator.class))
			.withConfiguration(AutoConfigurations.of(Inference4jAudioAutoConfiguration.class))
			.withPropertyValues(
					"inference4j.genai.text-generator.enabled=true",
					"inference4j.genai.whisper.enabled=true",
					"inference4j.audio.speech-recognizer.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx.getBeanNamesForType(SpeechRecognizer.class)).containsExactly("speechRecognizer");
				assertThat(ctx.containsBean("textGenerator")).isFalse();
			});
	}

	@Test
	void enabledBeansAreLazySoNoModelLoadsAtStartup() {
		runner.withPropertyValues(
				"inference4j.genai.text-generator.enabled=true",
				"inference4j.genai.vision-language.enabled=true",
				"inference4j.genai.whisper.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx.getBeanFactory().getBeanDefinition("textGenerator").isLazyInit()).isTrue();
				assertThat(ctx.getBeanFactory().getBeanDefinition("visionLanguageModel").isLazyInit()).isTrue();
				assertThat(ctx.getBeanFactory().getBeanDefinition("whisperSpeechModel").isLazyInit()).isTrue();
			});
	}

	@Test
	void whisperBecomesTheSpeechRecognizerInsteadOfWav2Vec2() {
		runner.withConfiguration(AutoConfigurations.of(Inference4jAudioAutoConfiguration.class))
			.withPropertyValues(
					"inference4j.audio.speech-recognizer.enabled=true",
					"inference4j.genai.whisper.enabled=true")
			.run(ctx -> assertThat(ctx.getBeanNamesForType(SpeechRecognizer.class))
				.containsExactly("whisperSpeechModel"));
	}

	@Test
	void wav2Vec2RemainsTheSpeechRecognizerWithoutWhisper() {
		runner.withConfiguration(AutoConfigurations.of(Inference4jAudioAutoConfiguration.class))
			.withPropertyValues("inference4j.audio.speech-recognizer.enabled=true")
			.run(ctx -> assertThat(ctx.getBeanNamesForType(SpeechRecognizer.class))
				.containsExactly("speechRecognizer"));
	}

	@Test
	void userBeanOverridesTheTextGenerator() {
		runner.withUserConfiguration(CustomTextGeneratorConfig.class)
			.withPropertyValues("inference4j.genai.text-generator.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(TextGenerator.class);
				assertThat(ctx).getBean(TextGenerator.class).isSameAs(CustomTextGeneratorConfig.INSTANCE);
			});
	}

	@Test
	void unknownPresetFailsWithTheSupportedNames() {
		runner.withPropertyValues(
				"inference4j.genai.text-generator.enabled=true",
				"inference4j.genai.text-generator.model=gpt-5")
			.run(ctx -> assertThatThrownBy(() -> ctx.getBean(TextGenerator.class))
				.rootCause()
				.isInstanceOf(IllegalArgumentException.class)
				.hasMessageContaining("gpt-5")
				.hasMessageContaining("deepseek-r1-1.5b")
				.hasMessageContaining("phi-3-mini"));
	}

	@Test
	void unknownWhisperTaskFailsWithTheSupportedTasks() {
		runner.withPropertyValues(
				"inference4j.genai.whisper.enabled=true",
				"inference4j.genai.whisper.task=summarize")
			.run(ctx -> assertThatThrownBy(() -> ctx.getBean(WhisperSpeechModel.class))
				.hasStackTraceContaining("Unknown inference4j.genai.whisper.task 'summarize'")
				.hasStackTraceContaining("transcribe, translate"));
	}

	@Configuration(proxyBeanMethods = false)
	static class CustomTextGeneratorConfig {

		static final TextGenerator INSTANCE = mock(TextGenerator.class);

		@Bean
		TextGenerator customTextGenerator() {
			return INSTANCE;
		}

	}

}
