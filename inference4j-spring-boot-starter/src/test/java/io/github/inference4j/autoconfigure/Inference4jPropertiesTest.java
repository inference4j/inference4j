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

import org.junit.jupiter.api.Test;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.*;

class Inference4jPropertiesTest {

    @Test
    void defaults_nlpTextClassifierDisabled() {
        Inference4jProperties properties = new Inference4jProperties();

        assertThat(properties.getNlp().getTextClassifier().isEnabled()).isFalse();
    }

    @Test
    void defaults_nlpTextClassifierDefaultModelId() {
        Inference4jProperties properties = new Inference4jProperties();

        assertThat(properties.getNlp().getTextClassifier().getModelId())
                .isEqualTo("inference4j/distilbert-base-uncased-finetuned-sst-2-english");
    }

    @Test
    void defaults_visionImageClassifierDefaultModelId() {
        Inference4jProperties properties = new Inference4jProperties();

        assertThat(properties.getVision().getImageClassifier().getModelId())
                .isEqualTo("inference4j/resnet50-v1-7");
    }

    @Test
    void defaultsModelsEndpointEnabled() {
        Inference4jProperties properties = new Inference4jProperties();

        assertThat(properties.getActuator().getModels().isEnabled()).isTrue();
    }

    @Test
    void defaultsVisionObjectDetectorIsDfineSmall() {
        Inference4jProperties properties = new Inference4jProperties();

        assertThat(properties.getVision().getObjectDetector().getArchitecture())
                .isEqualTo(Inference4jProperties.ObjectDetectorArchitecture.DETR);
        assertThat(properties.getVision().getObjectDetector().getModelId())
                .isEqualTo("inference4j/dfine-small-coco");
    }

    @Test
    void objectDetectorFallsBackToTheArchitectureDefaultModelId() {
        Inference4jProperties.ObjectDetectorProperties detector = new Inference4jProperties.ObjectDetectorProperties();

        detector.setArchitecture(Inference4jProperties.ObjectDetectorArchitecture.YOLOV8);
        assertThat(detector.getModelId()).isEqualTo("inference4j/yolov8n");

        detector.setArchitecture(Inference4jProperties.ObjectDetectorArchitecture.YOLO26);
        assertThat(detector.getModelId()).isEqualTo("inference4j/yolo26n");
    }

    @Test
    void objectDetectorKeepsAnExplicitModelIdAcrossArchitectures() {
        Inference4jProperties.ObjectDetectorProperties detector = new Inference4jProperties.ObjectDetectorProperties();

        detector.setModelId("inference4j/dfine-nano-coco");
        detector.setArchitecture(Inference4jProperties.ObjectDetectorArchitecture.YOLOV8);

        assertThat(detector.getModelId()).isEqualTo("inference4j/dfine-nano-coco");
    }

    @Test
    void objectDetectorPropertiesBindFromRelaxedNames() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropertiesConfig.class)
                .withPropertyValues(
                        "inference4j.vision.object-detector.enabled=true",
                        "inference4j.vision.object-detector.architecture=yolo26",
                        "inference4j.vision.object-detector.model-id=acme/my-yolo26")
                .run(ctx -> {
                    Inference4jProperties.ObjectDetectorProperties detector =
                            ctx.getBean(Inference4jProperties.class).getVision().getObjectDetector();
                    assertThat(detector.isEnabled()).isTrue();
                    assertThat(detector.getArchitecture())
                            .isEqualTo(Inference4jProperties.ObjectDetectorArchitecture.YOLO26);
                    assertThat(detector.getModelId()).isEqualTo("acme/my-yolo26");
                });
    }

    @Test
    void objectDetectorArchitectureBindsWithoutModelId() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropertiesConfig.class)
                .withPropertyValues("inference4j.vision.object-detector.architecture=yolov8")
                .run(ctx -> assertThat(ctx.getBean(Inference4jProperties.class)
                        .getVision().getObjectDetector().getModelId()).isEqualTo("inference4j/yolov8n"));
    }

    @Test
    void defaults_audioSpeechRecognizerDefaultModelId() {
        Inference4jProperties properties = new Inference4jProperties();

        assertThat(properties.getAudio().getSpeechRecognizer().getModelId())
                .isEqualTo("inference4j/wav2vec2-base-960h");
    }

    @Test
    void setEnabled_overridesDefault() {
        Inference4jProperties.TaskProperties task = new Inference4jProperties.TaskProperties();

        task.setEnabled(true);

        assertThat(task.isEnabled()).isTrue();
    }

    @Test
    void setModelId_overridesDefault() {
        Inference4jProperties.TaskProperties task = new Inference4jProperties.TaskProperties();

        task.setModelId("custom/model");

        assertThat(task.getModelId()).isEqualTo("custom/model");
    }

    @Test
    void nestedPropertyAccess_nlp() {
        Inference4jProperties properties = new Inference4jProperties();

        assertThat(properties.getNlp()).isNotNull();
        assertThat(properties.getNlp().getTextClassifier()).isNotNull();
        assertThat(properties.getNlp().getTextEmbedder()).isNotNull();
        assertThat(properties.getNlp().getSearchReranker()).isNotNull();
    }

    @Test
    void nestedPropertyAccess_vision() {
        Inference4jProperties properties = new Inference4jProperties();

        assertThat(properties.getVision()).isNotNull();
        assertThat(properties.getVision().getImageClassifier()).isNotNull();
        assertThat(properties.getVision().getObjectDetector()).isNotNull();
        assertThat(properties.getVision().getTextDetector()).isNotNull();
    }

    @Test
    void nestedPropertyAccess_audio() {
        Inference4jProperties properties = new Inference4jProperties();

        assertThat(properties.getAudio()).isNotNull();
        assertThat(properties.getAudio().getSpeechRecognizer()).isNotNull();
        assertThat(properties.getAudio().getVad()).isNotNull();
    }

    @Configuration(proxyBeanMethods = false)
    @EnableConfigurationProperties(Inference4jProperties.class)
    static class PropertiesConfig {
    }
}
