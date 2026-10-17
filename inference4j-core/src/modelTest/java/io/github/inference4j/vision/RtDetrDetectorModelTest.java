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

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Offset.offset;

/**
 * Expected detections are HuggingFace {@code transformers}' output ({@code RTDetrImageProcessor} pre- and
 * post-processing, threshold 0.5) for the PyTorch models the mirrors were made from.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class RtDetrDetectorModelTest {

    private BufferedImage cat;

    @BeforeAll
    void setUp() throws IOException {
        cat = ImageIO.read(RtDetrDetectorModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @Test
    void dfineSmallFindsTheCatLikeTheReference() {
        try (ObjectDetector detector = RtDetrDetector.builder().build()) {
            Detection top = detector.detect(cat).get(0);

            assertThat(top.label()).isEqualTo("cat");
            assertThat(top.confidence()).isCloseTo(0.9384f, offset(1e-3f));
            assertBox(top.box(), 0.1f, 0.5f, 259.0f, 318.5f);
        }
    }

    @Test
    void dfineNanoFindsTheCatLikeTheReference() {
        try (ObjectDetector detector = RtDetrDetector.dfineNano().build()) {
            List<Detection> detections = detector.detect(cat);

            assertThat(detections).singleElement().satisfies(d -> {
                assertThat(d.label()).isEqualTo("cat");
                assertThat(d.confidence()).isCloseTo(0.9536f, offset(1e-3f));
                assertBox(d.box(), 0.2f, 0.5f, 254.9f, 319.5f);
            });
        }
    }

    @Test
    void rtDetrV2FindsTheCatLikeTheReference() {
        try (ObjectDetector detector = RtDetrDetector.rtDetrV2().build()) {
            List<Detection> detections = detector.detect(cat);

            assertThat(detections).singleElement().satisfies(d -> {
                assertThat(d.label()).isEqualTo("cat");
                assertThat(d.confidence()).isCloseTo(0.9572f, offset(1e-3f));
                assertBox(d.box(), 0.2f, 0.7f, 254.6f, 317.8f);
            });
        }
    }

    @Test
    void aHigherThresholdDropsLessConfidentDetections() {
        try (var detector = RtDetrDetector.dfineSmall().build()) {
            assertThat(detector.detect(cat, 0.5f, 0f)).hasSizeGreaterThanOrEqualTo(1);
            assertThat(detector.detect(cat, 0.95f, 0f)).isEmpty();
        }
    }

    private static void assertBox(BoundingBox box, float x1, float y1, float x2, float y2) {
        assertThat(box.x1()).isCloseTo(x1, offset(1f));
        assertThat(box.y1()).isCloseTo(y1, offset(1f));
        assertThat(box.x2()).isCloseTo(x2, offset(1f));
        assertThat(box.y2()).isCloseTo(y2, offset(1f));
    }
}
