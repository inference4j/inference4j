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

import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.assertj.core.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class DepthAnythingEstimatorModelTest {

    private DepthAnythingEstimator estimator;
    private BufferedImage catImage;

    @BeforeAll
    void setUp() throws IOException {
        estimator = DepthAnythingEstimator.builder().build();
        catImage = ImageIO.read(
                DepthAnythingEstimatorModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
    }

    @AfterAll
    void tearDown() throws Exception {
        if (estimator != null) estimator.close();
    }

    @Test
    void estimate_catImage_matchesInputDimensions() {
        DepthMap depth = estimator.estimate(catImage);

        assertThat(depth.width()).isEqualTo(catImage.getWidth());
        assertThat(depth.height()).isEqualTo(catImage.getHeight());
        assertThat(depth.values().length).isEqualTo(catImage.getHeight());
        assertThat(depth.values()[0].length).isEqualTo(catImage.getWidth());
    }

    @Test
    void estimate_catImage_producesVaryingFiniteDepth() {
        DepthMap depth = estimator.estimate(catImage);

        for (float[] row : depth.values()) {
            for (float v : row) {
                assertThat(Float.isFinite(v)).as("Depth values should be finite, got: " + v).isTrue();
            }
        }
        assertThat(depth.max() > depth.min())
                .as("A real scene should produce varying depth, not a constant map")
                .isTrue();
    }

    @Test
    void estimate_catImage_rendersToImageOfSameSize() {
        DepthMap depth = estimator.estimate(catImage);
        BufferedImage rendered = depth.toImage(Colormap.TURBO);

        assertThat(rendered.getWidth()).isEqualTo(catImage.getWidth());
        assertThat(rendered.getHeight()).isEqualTo(catImage.getHeight());
    }
}
