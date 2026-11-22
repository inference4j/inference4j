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
import static org.assertj.core.data.Offset.offset;

/**
 * Expected detections come from PaddleDetection's pipeline for these models (bicubic stretch to 416 × 416, ImageNet
 * normalization, {@code PicoHeadV2} decode, class-aware NMS at IoU 0.6, score above 0.5), run on the published files.
 * The cropped image is not square, so it also checks the separate horizontal and vertical scales.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PicoDetDetectorModelTest {

    private static final SessionConfigurer FOUR_THREADS = options -> options.setIntraOpNumThreads(4);

    private BufferedImage cat;
    private BufferedImage croppedCat;

    @BeforeAll
    void setUp() throws IOException {
        cat = ImageIO.read(PicoDetDetectorModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
        croppedCat = cat.getSubimage(0, 32, 320, 256);
    }

    @Test
    void sFindsTheCatLikeTheReference() {
        try (ObjectDetector detector = PicoDetDetector.s().sessionOptions(FOUR_THREADS).build()) {
            assertThat(detector.detect(cat)).singleElement()
                    .satisfies(d -> assertCat(d, 0.9692f, 0.0f, 0.4f, 252.1f, 320.0f));
            assertThat(detector.detect(croppedCat)).singleElement()
                    .satisfies(d -> assertCat(d, 0.9709f, 0.0f, 0.0f, 252.6f, 256.0f));
        }
    }

    @Test
    void mFindsTheCatLikeTheReference() {
        try (ObjectDetector detector = PicoDetDetector.m().sessionOptions(FOUR_THREADS).build()) {
            assertThat(detector.detect(cat)).singleElement()
                    .satisfies(d -> assertCat(d, 0.9498f, 0.1f, 1.9f, 260.1f, 320.0f));
            assertThat(detector.detect(croppedCat)).singleElement()
                    .satisfies(d -> assertCat(d, 0.9585f, 0.0f, 0.0f, 269.4f, 256.0f));
        }
    }

    @Test
    void aHigherThresholdDropsLessConfidentDetections() {
        try (ObjectDetector detector = PicoDetDetector.s().sessionOptions(FOUR_THREADS).build()) {
            assertThat(detector.detect(cat, 0.5f)).hasSize(1);
            assertThat(detector.detect(cat, 0.98f)).isEmpty();
        }
    }

    @Test
    void inputSizeOtherThanTheFixedInputFails() {
        assertThatThrownBy(() -> PicoDetDetector.s().inputSize(640).build())
                .isInstanceOf(ModelLoadException.class)
                .hasMessageContaining("fixed 416 × 416 input");
    }

    private static void assertCat(Detection d, float confidence, float x1, float y1, float x2, float y2) {
        assertThat(d.label()).isEqualTo("cat");
        assertThat(d.classIndex()).isEqualTo(15);
        // ImageIO decodes cat.jpg to the same pixels as OpenCV; boxes are clamped to the image
        assertThat(d.confidence()).isCloseTo(confidence, offset(1e-3f));
        assertThat(d.box().x1()).isCloseTo(x1, offset(1f));
        assertThat(d.box().y1()).isCloseTo(y1, offset(1f));
        assertThat(d.box().x2()).isCloseTo(x2, offset(1f));
        assertThat(d.box().y2()).isCloseTo(y2, offset(1f));
    }
}
