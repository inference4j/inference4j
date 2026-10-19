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

import java.awt.Color;
import java.awt.image.BufferedImage;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class SegmentationMaskTest {

    /** A 4x3 mask with a 2x2 square at columns 1-2, rows 1-2. */
    private static SegmentationMask square() {
        boolean[][] mask = new boolean[3][4];
        mask[1][1] = true;
        mask[1][2] = true;
        mask[2][1] = true;
        mask[2][2] = true;
        return new SegmentationMask(mask, 4, 3, 0.9f);
    }

    private static BufferedImage solid(int width, int height, int rgb) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.setRGB(x, y, rgb);
            }
        }
        return image;
    }

    @Test
    void containsReadsPixelMembership() {
        assertThat(square().contains(1, 1)).isTrue();
        assertThat(square().contains(0, 0)).isFalse();
    }

    @Test
    void areaCountsMaskedPixels() {
        assertThat(square().area()).isEqualTo(4);
        assertThat(square().isEmpty()).isFalse();
    }

    @Test
    void boundingBoxIsTightAndExclusive() {
        assertThat(square().boundingBox()).contains(new BoundingBox(1, 1, 3, 3));
    }

    @Test
    void emptyMaskHasNoBoundingBox() {
        SegmentationMask empty = new SegmentationMask(new boolean[2][2], 2, 2, 0f);

        assertThat(empty.isEmpty()).isTrue();
        assertThat(empty.boundingBox()).isEmpty();
    }

    @Test
    void toImageIsWhiteInsideAndBlackOutside() {
        BufferedImage image = square().toImage();

        assertThat(image.getWidth()).isEqualTo(4);
        assertThat(image.getHeight()).isEqualTo(3);
        assertThat(image.getRGB(1, 1) & 0xFFFFFF).isEqualTo(0xFFFFFF);
        assertThat(image.getRGB(0, 0) & 0xFFFFFF).isZero();
    }

    @Test
    void applyToKeepsMaskedPixelsAndClearsTheRest() {
        BufferedImage cutout = square().applyTo(solid(4, 3, 0x336699));

        assertThat(cutout.getRGB(1, 1)).isEqualTo(0xFF336699);
        assertThat(cutout.getRGB(0, 0) >>> 24).isZero();
    }

    @Test
    void overlayBlendsColourOnlyInsideMask() {
        BufferedImage result = square().overlay(solid(4, 3, 0x000000), Color.WHITE);

        assertThat(result.getRGB(1, 1) & 0xFFFFFF).isEqualTo(0x808080);
        assertThat(result.getRGB(0, 0) & 0xFFFFFF).isZero();
    }

    @Test
    void applyToRejectsImageOfDifferentSize() {
        assertThatThrownBy(() -> square().applyTo(solid(5, 3, 0)))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("5x3");
    }
}
