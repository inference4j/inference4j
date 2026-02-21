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

class CraftTextDetectorModelTest {

    private static BufferedImage loadTextImage() throws IOException {
        return ImageIO.read(CraftTextDetectorModelTest.class.getResourceAsStream("/fixtures/text.jpg"));
    }

    @Test
    void detect_textImage_findsTextRegions() throws IOException {
        try (var detector = CraftTextDetector.builder()
                .textThreshold(0.4f)
                .lowTextThreshold(0.3f)
                .build()) {
            List<TextRegion> regions = detector.detect(loadTextImage());

            assertFalse(regions.isEmpty(), "Should detect at least one text region");
        }
    }

    @Test
    void detect_textImage_regionsHaveValidCoordinates() throws IOException {
        try (var detector = CraftTextDetector.builder()
                .textThreshold(0.4f)
                .lowTextThreshold(0.3f)
                .build()) {
            List<TextRegion> regions = detector.detect(loadTextImage());

            for (TextRegion region : regions) {
                BoundingBox box = region.box();
                assertTrue(box.x1() >= 0, "x1 should be >= 0, got: " + box.x1());
                assertTrue(box.y1() >= 0, "y1 should be >= 0, got: " + box.y1());
                assertTrue(box.x2() > box.x1(), "x2 should be > x1");
                assertTrue(box.y2() > box.y1(), "y2 should be > y1");
                assertTrue(region.confidence() > 0f, "Confidence should be positive");
            }
        }
    }
}
