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
 * Expected detections come from the official YOLOX ONNX Runtime demo ({@code preproc}, {@code demo_postprocess},
 * class-agnostic {@code multiclass_nms}, score 0.3, IoU 0.45) on the same files. The cropped image is not square, so
 * it also checks the top-left padding and the mapping back to the image.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class YoloXDetectorModelTest {

    private static final SessionConfigurer FOUR_THREADS = options -> options.setIntraOpNumThreads(4);

    private BufferedImage cat;
    private BufferedImage croppedCat;

    @BeforeAll
    void setUp() throws IOException {
        cat = ImageIO.read(YoloXDetectorModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
        croppedCat = cat.getSubimage(0, 32, 320, 256);
    }

    @Test
    void nanoFindsTheCatLikeTheReference() {
        try (ObjectDetector detector = YoloXDetector.nano().sessionOptions(FOUR_THREADS).build()) {
            assertSingleCat(detector.detect(cat).get(0), 0.7157f, 1.5f, 29.5f, 291.6f, 290.2f);
            assertSingleCat(detector.detect(croppedCat).get(0), 0.7095f, 3.7f, 3.9f, 289.2f, 248.0f);
            assertThat(detector.detect(cat)).hasSize(1);
        }
    }

    @Test
    void tinyFindsTheCatLikeTheReference() {
        try (ObjectDetector detector = YoloXDetector.tiny().sessionOptions(FOUR_THREADS).build()) {
            assertSingleCat(detector.detect(cat).get(0), 0.8536f, 7.5f, 18.1f, 263.5f, 309.4f);
            assertSingleCat(detector.detect(croppedCat).get(0), 0.8857f, 7.1f, 5.3f, 263.7f, 250.2f);
            assertThat(detector.detect(cat)).hasSize(1);
        }
    }

    @Test
    void sFindsTheCatLikeTheReference() {
        try (ObjectDetector detector = YoloXDetector.s().sessionOptions(FOUR_THREADS).build()) {
            assertSingleCat(detector.detect(cat).get(0), 0.8960f, 1.7f, 5.4f, 259.5f, 314.6f);
            assertSingleCat(detector.detect(croppedCat).get(0), 0.9296f, 2.1f, 0.8f, 258.1f, 252.9f);
            assertThat(detector.detect(cat)).hasSize(1);
        }
    }

    @Test
    void aHigherThresholdDropsLessConfidentDetections() {
        try (ObjectDetector detector = YoloXDetector.nano().sessionOptions(FOUR_THREADS).build()) {
            assertThat(detector.detect(cat, 0.3f)).hasSize(1);
            assertThat(detector.detect(cat, 0.8f)).isEmpty();
        }
    }

    @Test
    void inputSizeOtherThanTheFixedInputFails() {
        assertThatThrownBy(() -> YoloXDetector.nano().inputSize(640).build())
                .isInstanceOf(ModelLoadException.class)
                .hasMessageContaining("fixed 416 × 416 input");
    }

    private static void assertSingleCat(Detection d, float confidence, float x1, float y1, float x2, float y2) {
        assertThat(d.label()).isEqualTo("cat");
        assertThat(d.classIndex()).isEqualTo(15);
        // The reference ran OpenCV 5.1, whose bilinear resize moves scores in the third decimal against OpenCV 4.x
        assertThat(d.confidence()).isCloseTo(confidence, offset(5e-3f));
        assertThat(d.box().x1()).isCloseTo(x1, offset(1f));
        assertThat(d.box().y1()).isCloseTo(y1, offset(1f));
        assertThat(d.box().x2()).isCloseTo(x2, offset(1f));
        assertThat(d.box().y2()).isCloseTo(y2, offset(1f));
    }
}
