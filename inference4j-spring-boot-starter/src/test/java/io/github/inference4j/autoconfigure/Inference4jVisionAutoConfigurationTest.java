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

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

import io.github.inference4j.vision.Classification;
import io.github.inference4j.vision.Detection;
import io.github.inference4j.vision.ImageClassifier;
import io.github.inference4j.vision.ObjectDetector;
import io.github.inference4j.vision.TextDetector;
import io.github.inference4j.vision.TextRegion;
import org.junit.jupiter.api.Test;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;

class Inference4jVisionAutoConfigurationTest {

	private final ApplicationContextRunner runner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(Inference4jVisionAutoConfiguration.class));

	@Test
	void noBeansWhenDisabled() {
		runner.run(ctx -> {
			assertThat(ctx).doesNotHaveBean(ImageClassifier.class);
			assertThat(ctx).doesNotHaveBean(ObjectDetector.class);
			assertThat(ctx).doesNotHaveBean(TextDetector.class);
		});
	}

	@Test
	void noBeansWhenEnabledSetToFalse() {
		runner.withPropertyValues(
				"inference4j.vision.image-classifier.enabled=false",
				"inference4j.vision.object-detector.enabled=false",
				"inference4j.vision.text-detector.enabled=false")
			.run(ctx -> {
				assertThat(ctx).doesNotHaveBean(ImageClassifier.class);
				assertThat(ctx).doesNotHaveBean(ObjectDetector.class);
				assertThat(ctx).doesNotHaveBean(TextDetector.class);
			});
	}

	@Test
	void userBeanOverridesAutoConfiguredImageClassifier() {
		runner.withPropertyValues("inference4j.vision.image-classifier.enabled=true")
			.withUserConfiguration(CustomImageClassifierConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(ImageClassifier.class);
				assertThat(ctx).getBean(ImageClassifier.class).isSameAs(CustomImageClassifierConfig.INSTANCE);
			});
	}

	@Test
	void userBeanOverridesAutoConfiguredObjectDetector() {
		runner.withPropertyValues("inference4j.vision.object-detector.enabled=true")
			.withUserConfiguration(CustomObjectDetectorConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(ObjectDetector.class);
				assertThat(ctx).getBean(ObjectDetector.class).isSameAs(CustomObjectDetectorConfig.INSTANCE);
			});
	}

	@Test
	void userBeanOverridesAutoConfiguredTextDetector() {
		runner.withPropertyValues("inference4j.vision.text-detector.enabled=true")
			.withUserConfiguration(CustomTextDetectorConfig.class)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(TextDetector.class);
				assertThat(ctx).getBean(TextDetector.class).isSameAs(CustomTextDetectorConfig.INSTANCE);
			});
	}

	@Configuration(proxyBeanMethods = false)
	static class CustomImageClassifierConfig {

		static final ImageClassifier INSTANCE = new ImageClassifier() {
			@Override
			public List<Classification> classify(BufferedImage image) {
				return List.of();
			}

			@Override
			public List<Classification> classify(BufferedImage image, int topK) {
				return List.of();
			}

			@Override
			public List<Classification> classify(Path imagePath) {
				return List.of();
			}

			@Override
			public List<Classification> classify(Path imagePath, int topK) {
				return List.of();
			}

			@Override
			public void close() {
			}
		};

		@Bean
		ImageClassifier imageClassifier() {
			return INSTANCE;
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class CustomObjectDetectorConfig {

		static final ObjectDetector INSTANCE = new ObjectDetector() {
			@Override
			public List<Detection> detect(BufferedImage image) {
				return List.of();
			}

			@Override
			public List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold) {
				return List.of();
			}

			@Override
			public List<Detection> detect(Path imagePath) {
				return List.of();
			}

			@Override
			public List<Detection> detect(Path imagePath, float confidenceThreshold, float iouThreshold) {
				return List.of();
			}

			@Override
			public void close() {
			}
		};

		@Bean
		ObjectDetector objectDetector() {
			return INSTANCE;
		}

	}

	@Configuration(proxyBeanMethods = false)
	static class CustomTextDetectorConfig {

		static final TextDetector INSTANCE = new TextDetector() {
			@Override
			public List<TextRegion> detect(BufferedImage image) {
				return List.of();
			}

			@Override
			public List<TextRegion> detect(BufferedImage image, float textThreshold, float lowTextThreshold) {
				return List.of();
			}

			@Override
			public List<TextRegion> detect(Path imagePath) {
				return List.of();
			}

			@Override
			public List<TextRegion> detect(Path imagePath, float textThreshold, float lowTextThreshold) {
				return List.of();
			}

			@Override
			public void close() {
			}
		};

		@Bean
		TextDetector textDetector() {
			return INSTANCE;
		}

	}

}
