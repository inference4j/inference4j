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

import io.github.inference4j.audio.SileroVadDetector;
import io.github.inference4j.audio.SpeechRecognizer;
import io.github.inference4j.audio.VoiceActivityDetector;
import io.github.inference4j.audio.Wav2Vec2Recognizer;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Auto-configuration for inference4j audio tasks.
 */
@AutoConfiguration
@ConditionalOnClass(Wav2Vec2Recognizer.class)
@EnableConfigurationProperties(Inference4jProperties.class)
public class Inference4jAudioAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean(SpeechRecognizer.class)
	@ConditionalOnProperty(prefix = "inference4j.audio.speech-recognizer", name = "enabled", havingValue = "true")
	public SpeechRecognizer speechRecognizer(Inference4jProperties properties) {
		return Wav2Vec2Recognizer.builder()
			.modelId(properties.getAudio().getSpeechRecognizer().getModelId())
			.build();
	}

	@Bean
	@ConditionalOnMissingBean(VoiceActivityDetector.class)
	@ConditionalOnProperty(prefix = "inference4j.audio.vad", name = "enabled", havingValue = "true")
	public VoiceActivityDetector voiceActivityDetector(Inference4jProperties properties) {
		return SileroVadDetector.builder()
			.modelId(properties.getAudio().getVad().getModelId())
			.build();
	}

}
