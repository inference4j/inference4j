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
}
