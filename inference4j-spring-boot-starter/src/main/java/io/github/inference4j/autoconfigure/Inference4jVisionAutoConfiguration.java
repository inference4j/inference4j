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

import io.github.inference4j.vision.CraftTextDetector;
import io.github.inference4j.vision.ImageClassifier;
import io.github.inference4j.vision.ObjectDetector;
import io.github.inference4j.vision.ResNetClassifier;
import io.github.inference4j.vision.TextDetector;
import io.github.inference4j.vision.YoloV8Detector;

import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Lazy;

/**
 * Auto-configuration for inference4j vision tasks.
 */
@AutoConfiguration
@ConditionalOnClass(ResNetClassifier.class)
@EnableConfigurationProperties(Inference4jProperties.class)
public class Inference4jVisionAutoConfiguration {

	@Bean
	@Lazy
	@ConditionalOnMissingBean(ImageClassifier.class)
	@ConditionalOnProperty(prefix = "inference4j.vision.image-classifier", name = "enabled", havingValue = "true")
	public ImageClassifier imageClassifier(Inference4jProperties properties) {
		return ResNetClassifier.builder()
			.modelId(properties.getVision().getImageClassifier().getModelId())
			.build();
	}

	@Bean
	@Lazy
	@ConditionalOnMissingBean(ObjectDetector.class)
	@ConditionalOnProperty(prefix = "inference4j.vision.object-detector", name = "enabled", havingValue = "true")
	public ObjectDetector objectDetector(Inference4jProperties properties) {
		return YoloV8Detector.builder()
			.modelId(properties.getVision().getObjectDetector().getModelId())
			.build();
	}

	@Bean
	@Lazy
	@ConditionalOnMissingBean(TextDetector.class)
	@ConditionalOnProperty(prefix = "inference4j.vision.text-detector", name = "enabled", havingValue = "true")
	public TextDetector textDetector(Inference4jProperties properties) {
		return CraftTextDetector.builder()
			.modelId(properties.getVision().getTextDetector().getModelId())
			.build();
	}

}
