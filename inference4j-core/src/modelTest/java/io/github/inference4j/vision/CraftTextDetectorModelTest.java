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
class CraftTextDetectorModelTest {

    private CraftTextDetector detector;
    private BufferedImage textImage;

    @BeforeAll
    void setUp() throws IOException {
        detector = CraftTextDetector.builder()
                .textThreshold(0.4f)
                .lowTextThreshold(0.3f)
                .build();
        textImage = ImageIO.read(CraftTextDetectorModelTest.class.getResourceAsStream("/fixtures/text.jpg"));
    }

    @AfterAll
    void tearDown() throws Exception {
        if (detector != null) detector.close();
    }

    @Test
    void detect_textImage_findsTextRegions() {
        List<TextRegion> regions = detector.detect(textImage);

        assertThat(regions.isEmpty()).as("Should detect at least one text region").isFalse();
    }

    @Test
    void detect_textImage_regionsHaveValidCoordinates() {
        List<TextRegion> regions = detector.detect(textImage);

        for (TextRegion region : regions) {
            BoundingBox box = region.box();
            assertThat(box.x1() >= 0).as("x1 should be >= 0, got: " + box.x1()).isTrue();
            assertThat(box.y1() >= 0).as("y1 should be >= 0, got: " + box.y1()).isTrue();
            assertThat(box.x2() > box.x1()).as("x2 should be > x1").isTrue();
            assertThat(box.y2() > box.y1()).as("y2 should be > y1").isTrue();
            assertThat(region.confidence() > 0f).as("Confidence should be positive").isTrue();
        }
    }
}
