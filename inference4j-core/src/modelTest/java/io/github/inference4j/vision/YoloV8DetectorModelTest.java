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

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class YoloV8DetectorModelTest {

    private static BufferedImage loadCatImage() throws IOException {
        return ImageIO.read(YoloV8DetectorModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @Test
    void detect_catImage_returnsDetections() throws IOException {
        try (var detector = YoloV8Detector.builder().build()) {
            List<Detection> detections = detector.detect(loadCatImage());

            assertFalse(detections.isEmpty(), "Should detect at least one object in cat image");
        }
    }

    @Test
    void detect_catImage_detectionsHaveValidFields() throws IOException {
        try (var detector = YoloV8Detector.builder().build()) {
            List<Detection> detections = detector.detect(loadCatImage());

            for (Detection d : detections) {
                assertNotNull(d.label(), "Detection label should not be null");
                assertFalse(d.label().isBlank(), "Detection label should not be blank");
                assertTrue(d.confidence() > 0f, "Confidence should be positive, got: " + d.confidence());
                assertTrue(d.confidence() <= 1f, "Confidence should be <= 1, got: " + d.confidence());

                BoundingBox box = d.box();
                assertTrue(box.x2() > box.x1(), "Box x2 should be > x1");
                assertTrue(box.y2() > box.y1(), "Box y2 should be > y1");
                assertTrue(box.area() > 0, "Box area should be positive");
            }
        }
    }
}
