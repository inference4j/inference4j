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

import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.session.SessionConfigurer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Loads the RTMO presets and runs them on an image without people. The export appends a padding row (score 0) that
 * must never come back as a person, even with a threshold of 0; real low-score rows (RTMO-s scores the cat's face
 * 0.21) are kept above the model's own 0.1. Keypoint parity against the exporter's reference (MMDeploy ONNX through
 * OpenCV pre-processing) was checked on seven COCO val2017 images when the presets were added: same people, boxes
 * within 0.4 px, person scores within 1e-4, confident keypoints within 1.4 px.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class RtmoPoseDetectorModelTest {

    private static final SessionConfigurer FOUR_THREADS = options -> options.setIntraOpNumThreads(4);

    private BufferedImage cat;

    @BeforeAll
    void setUp() throws IOException {
        cat = ImageIO.read(RtmoPoseDetectorModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @Test
    void sFindsNobodyInAnImageWithoutPeople() {
        assertNobody(RtmoPoseDetector.s());
    }

    @Test
    void mFindsNobodyInAnImageWithoutPeople() {
        assertNobody(RtmoPoseDetector.m());
    }

    @Test
    void lFindsNobodyInAnImageWithoutPeople() {
        assertNobody(RtmoPoseDetector.l());
    }

    @Test
    void inputSizeOtherThanTheFixedInputFails() {
        assertThatThrownBy(() -> RtmoPoseDetector.s().inputSize(416).build())
                .isInstanceOf(ModelLoadException.class)
                .hasMessageContaining("fixed 640 × 640 input");
    }

    private void assertNobody(RtmoPoseDetector.Builder builder) {
        try (PoseDetector detector = builder.sessionOptions(FOUR_THREADS).build()) {
            assertThat(detector.detect(cat)).isEmpty();
            assertThat(detector.detect(cat, 0f)).allSatisfy(pose -> {
                assertThat(pose.person().confidence()).isGreaterThan(0.1f);
                assertThat(pose.keypoints()).hasSize(17);
            });
        }
    }
}
