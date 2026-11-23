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
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.vision.Classification;
import io.github.inference4j.vision.Detection;
import io.github.inference4j.vision.DetrDetector;
import io.github.inference4j.vision.ImageClassifier;
import io.github.inference4j.vision.ObjectDetector;
import io.github.inference4j.vision.Pose;
import io.github.inference4j.vision.PoseDetector;
import io.github.inference4j.vision.RtmoPoseDetector;
import io.github.inference4j.vision.TextDetector;
import io.github.inference4j.vision.TextRegion;
import io.github.inference4j.vision.Yolo26Detector;
import io.github.inference4j.vision.YoloV8Detector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class Inference4jVisionAutoConfigurationTest {

	private final ApplicationContextRunner runner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(Inference4jVisionAutoConfiguration.class));

	/**
	 * Seeds the model cache that the build points {@code inference4j.cache.dir} at: a tiny stub model under
	 * each default id, and an unreadable model under the custom ids, so no test downloads anything.
	 */
	@BeforeAll
	static void seedModelCache() throws IOException {
		Path cache = Path.of(System.getProperty("inference4j.cache.dir"));
		for (String modelId : List.of("inference4j/dfine-small-coco", "inference4j/yolov8n", "inference4j/yolo26n")) {
			Path dir = Files.createDirectories(cache.resolve(modelId));
			try (InputStream stub = Inference4jVisionAutoConfigurationTest.class
				.getResourceAsStream("/detector-stub.onnx")) {
				Files.copy(stub, dir.resolve("model.onnx"), StandardCopyOption.REPLACE_EXISTING);
			}
		}
		Path pose = Files.createDirectories(cache.resolve("inference4j/rtmo-s-coco"));
		try (InputStream stub = Inference4jVisionAutoConfigurationTest.class.getResourceAsStream("/pose-stub.onnx")) {
			Files.copy(stub, pose.resolve("model.onnx"), StandardCopyOption.REPLACE_EXISTING);
		}
		for (String modelId : List.of("acme/custom-detr", "acme/custom-yolov8", "acme/custom-yolo26",
				"acme/custom-rtmo")) {
			Path dir = Files.createDirectories(cache.resolve(modelId));
			Files.writeString(dir.resolve("model.onnx"), "not an onnx model", StandardCharsets.UTF_8);
		}
	}

	@Test
	void noBeansWhenDisabled() {
		runner.run(ctx -> {
			assertThat(ctx).doesNotHaveBean(ImageClassifier.class);
			assertThat(ctx).doesNotHaveBean(ObjectDetector.class);
			assertThat(ctx).doesNotHaveBean(TextDetector.class);
			assertThat(ctx).doesNotHaveBean(PoseDetector.class);
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
	void defaultObjectDetectorIsDfineSmall() {
		runner.withPropertyValues("inference4j.vision.object-detector.enabled=true")
			.run(ctx -> assertThat(ctx.getBean(ObjectDetector.class)).isInstanceOf(DetrDetector.class));
	}

	@Test
	void detrArchitectureBuildsDetrDetector() {
		runner.withPropertyValues(
				"inference4j.vision.object-detector.enabled=true",
				"inference4j.vision.object-detector.architecture=detr")
			.run(ctx -> assertThat(ctx.getBean(ObjectDetector.class)).isInstanceOf(DetrDetector.class));
	}

	@Test
	void yolov8ArchitectureBuildsYoloV8DetectorWithItsOwnDefaultModel() {
		runner.withPropertyValues(
				"inference4j.vision.object-detector.enabled=true",
				"inference4j.vision.object-detector.architecture=yolov8")
			.run(ctx -> assertThat(ctx.getBean(ObjectDetector.class)).isInstanceOf(YoloV8Detector.class));
	}

	@Test
	void yolo26ArchitectureBuildsYolo26DetectorWithItsOwnDefaultModel() {
		runner.withPropertyValues(
				"inference4j.vision.object-detector.enabled=true",
				"inference4j.vision.object-detector.architecture=yolo26")
			.run(ctx -> assertThat(ctx.getBean(ObjectDetector.class)).isInstanceOf(Yolo26Detector.class));
	}

	@Test
	void detrArchitectureLoadsTheConfiguredModelId() {
		assertLoadsConfiguredModel("detr", "acme/custom-detr", "DetrDetector$Builder");
	}

	@Test
	void yolov8ArchitectureLoadsTheConfiguredModelId() {
		assertLoadsConfiguredModel("yolov8", "acme/custom-yolov8", "YoloV8Detector$Builder");
	}

	@Test
	void yolo26ArchitectureLoadsTheConfiguredModelId() {
		assertLoadsConfiguredModel("yolo26", "acme/custom-yolo26", "Yolo26Detector$Builder");
	}

	@Test
	void objectDetectorBeanIsLazy() {
		runner.withPropertyValues("inference4j.vision.object-detector.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx.getBeanFactory().getBeanDefinition("objectDetector").isLazyInit()).isTrue();
			});
	}

	private void assertLoadsConfiguredModel(String architecture, String modelId, String builder) {
		runner.withPropertyValues(
				"inference4j.vision.object-detector.enabled=true",
				"inference4j.vision.object-detector.architecture=" + architecture,
				"inference4j.vision.object-detector.model-id=" + modelId)
			.run(ctx -> assertThatThrownBy(() -> ctx.getBean(ObjectDetector.class))
				.hasStackTraceContaining(ModelLoadException.class.getName() + ": Failed to load model from")
				// The flat-layout stub, adopted into the repo's snapshot directory
				.hasStackTraceContaining("models--" + modelId.replace("/", "--"))
				.hasStackTraceContaining(builder));
	}

	@Test
	void defaultPoseDetectorIsRtmoS() {
		runner.withPropertyValues("inference4j.vision.pose-detector.enabled=true")
			.run(ctx -> assertThat(ctx.getBean(PoseDetector.class)).isInstanceOf(RtmoPoseDetector.class));
	}

	@Test
	void poseDetectorAppliesTheConfiguredThreshold() {
		// The stub scores its people 0.9 and 0.8 above the default threshold of 0.5
		BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
		runner.withPropertyValues("inference4j.vision.pose-detector.enabled=true")
			.run(ctx -> assertThat(ctx.getBean(PoseDetector.class).detect(image)).hasSize(2));
		runner.withPropertyValues("inference4j.vision.pose-detector.enabled=true",
				"inference4j.vision.pose-detector.confidence-threshold=0.85")
			.run(ctx -> assertThat(ctx.getBean(PoseDetector.class).detect(image)).hasSize(1));
	}

	@Test
	void poseDetectorLoadsTheConfiguredModelId() {
		runner.withPropertyValues("inference4j.vision.pose-detector.enabled=true",
				"inference4j.vision.pose-detector.model-id=acme/custom-rtmo")
			.run(ctx -> assertThatThrownBy(() -> ctx.getBean(PoseDetector.class))
				.hasStackTraceContaining(ModelLoadException.class.getName() + ": Failed to load model from")
				.hasStackTraceContaining("models--acme--custom-rtmo")
				.hasStackTraceContaining("RtmoPoseDetector$Builder"));
	}

	@Test
	void poseDetectorBeanIsLazy() {
		runner.withPropertyValues("inference4j.vision.pose-detector.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx.getBeanFactory().getBeanDefinition("poseDetector").isLazyInit()).isTrue();
			});
	}

	@Test
	void userBeanOverridesAutoConfiguredPoseDetector() {
		PoseDetector custom = new PoseDetector() {
			@Override
			public List<Pose> detect(BufferedImage image) {
				return List.of();
			}

			@Override
			public List<Pose> detect(BufferedImage image, float confidenceThreshold) {
				return List.of();
			}

			@Override
			public void close() {
			}
		};
		runner.withPropertyValues("inference4j.vision.pose-detector.enabled=true")
			.withBean(PoseDetector.class, () -> custom)
			.run(ctx -> {
				assertThat(ctx).hasSingleBean(PoseDetector.class);
				assertThat(ctx).getBean(PoseDetector.class).isSameAs(custom);
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
			public List<TextRegion> detect(Path imagePath) {
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
