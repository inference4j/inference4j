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

import java.nio.file.Path;
import java.util.List;

import io.github.inference4j.audio.SpeechRecognizer;
import io.github.inference4j.audio.Transcription;
import io.github.inference4j.audio.VoiceActivityDetector;
import io.github.inference4j.audio.VoiceSegment;
import org.junit.jupiter.api.Test;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;

class Inference4jAudioAutoConfigurationTest {

	private final ApplicationContextRunner runner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(Inference4jAudioAutoConfiguration.class));

	@Test
	void noBeansWhenDisabled() {
		runner.run(ctx -> {
			assertThat(ctx).doesNotHaveBean(SpeechRecognizer.class);
			assertThat(ctx).doesNotHaveBean(VoiceActivityDetector.class);
		});
	}

	@Test
	void noBeansWhenEnabledSetToFalse() {
		runner.withPropertyValues(
				"inference4j.audio.speech-recognizer.enabled=false",
				"inference4j.audio.vad.enabled=false")
			.run(ctx -> {
				assertThat(ctx).doesNotHaveBean(SpeechRecognizer.class);
				assertThat(ctx).doesNotHaveBean(VoiceActivityDetector.class);
			});
	}

	@Test
	void userBeanOverridesAutoConfiguredSpeechRecognizer() {
		runner.withPropertyValues("inference4j.audio.speech-recognizer.enabled=true")
			.withUserConfiguration(CustomSpeechRecognizerConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(SpeechRecognizer.class);
				assertThat(ctx).getBean(SpeechRecognizer.class).isSameAs(CustomSpeechRecognizerConfig.INSTANCE);
			});
	}

	@Test
	void userBeanOverridesAutoConfiguredVad() {
		runner.withPropertyValues("inference4j.audio.vad.enabled=true")
			.withUserConfiguration(CustomVadConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(VoiceActivityDetector.class);
				assertThat(ctx).getBean(VoiceActivityDetector.class).isSameAs(CustomVadConfig.INSTANCE);
			});
	}

	@Configuration(proxyBeanMethods = false)
	static class CustomSpeechRecognizerConfig {

		static final SpeechRecognizer INSTANCE = new SpeechRecognizer() {
			@Override
			public Transcription transcribe(Path audioPath) {
				return null;
			}

			@Override
			public Transcription transcribe(float[] audioData, int sampleRate) {
				return null;
			}

			@Override
			public void close() {
			}
		};

		@Bean
		SpeechRecognizer speechRecognizer() {
			return INSTANCE;
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class CustomVadConfig {

		static final VoiceActivityDetector INSTANCE = new VoiceActivityDetector() {
			@Override
			public List<VoiceSegment> detect(Path audioPath) {
				return List.of();
			}

			@Override
			public List<VoiceSegment> detect(float[] audioData, int sampleRate) {
				return List.of();
			}

			@Override
			public void close() {
			}
		};

		@Bean
		VoiceActivityDetector voiceActivityDetector() {
			return INSTANCE;
		}

	}

}
