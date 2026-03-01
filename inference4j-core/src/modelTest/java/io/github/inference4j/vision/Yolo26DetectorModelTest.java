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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class Yolo26DetectorModelTest {

    private Yolo26Detector detector;
    private BufferedImage catImage;

    @BeforeAll
    void setUp() throws IOException {
        detector = Yolo26Detector.builder().build();
        catImage = ImageIO.read(Yolo26DetectorModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @AfterAll
    void tearDown() throws Exception {
        if (detector != null) detector.close();
    }

    @Test
    void detect_catImage_returnsDetections() {
        List<Detection> detections = detector.detect(catImage);

        assertThat(detections.isEmpty()).as("Should detect at least one object in cat image").isFalse();
    }

    @Test
    void detect_catImage_detectionsHaveValidFields() {
        List<Detection> detections = detector.detect(catImage);

        for (Detection d : detections) {
            assertThat(d.label()).as("Detection label should not be null").isNotNull();
            assertThat(d.label().isBlank()).as("Detection label should not be blank").isFalse();
            assertThat(d.confidence() > 0f).as("Confidence should be positive, got: " + d.confidence()).isTrue();
            assertThat(d.confidence() <= 1f).as("Confidence should be <= 1, got: " + d.confidence()).isTrue();

            BoundingBox box = d.box();
            assertThat(box.x2() > box.x1()).as("Box x2 should be > x1").isTrue();
            assertThat(box.y2() > box.y1()).as("Box y2 should be > y1").isTrue();
            assertThat(box.area() > 0).as("Box area should be positive").isTrue();
        }
    }
}
