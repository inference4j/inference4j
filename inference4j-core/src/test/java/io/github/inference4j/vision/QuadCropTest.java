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

import java.awt.image.BufferedImage;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class QuadCropTest {

    @Test
    void anAxisAlignedRegionIsCopiedPixelForPixel() {
        BufferedImage image = pattern(40, 30);

        BufferedImage crop = QuadCrop.crop(image, rectangle(5, 7, 25, 17));

        assertThat(crop.getWidth()).isEqualTo(20);
        assertThat(crop.getHeight()).isEqualTo(10);
        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 20; x++) {
                assertThat(crop.getRGB(x, y) & 0xFFFFFF).isEqualTo(image.getRGB(5 + x, 7 + y) & 0xFFFFFF);
            }
        }
    }

    @Test
    void aTallRegionIsTurnedCounterClockwise() {
        BufferedImage image = pattern(40, 40);

        BufferedImage crop = QuadCrop.crop(image, rectangle(10, 5, 16, 35));   // 6 wide, 30 tall

        assertThat(crop.getWidth()).isEqualTo(30);
        assertThat(crop.getHeight()).isEqualTo(6);
        // Turning counter-clockwise puts the region's top-right pixel at the crop's top-left
        assertThat(crop.getRGB(0, 0) & 0xFFFFFF).isEqualTo(image.getRGB(15, 5) & 0xFFFFFF);
        assertThat(crop.getRGB(29, 0) & 0xFFFFFF).isEqualTo(image.getRGB(15, 34) & 0xFFFFFF);
    }

    @Test
    void cornerOrderDoesNotMatter() {
        BufferedImage image = pattern(40, 30);
        List<Point> shuffled = List.of(new Point(25, 17), new Point(5, 7), new Point(5, 17), new Point(25, 7));

        BufferedImage crop = QuadCrop.crop(image, shuffled);

        assertThat(crop.getRGB(0, 0) & 0xFFFFFF).isEqualTo(image.getRGB(5, 7) & 0xFFFFFF);
    }

    @Test
    void needsAtLeastThreePoints() {
        assertThatThrownBy(() -> QuadCrop.crop(pattern(4, 4), List.of(new Point(0, 0), new Point(1, 1))))
                .isInstanceOf(IllegalArgumentException.class);
    }

    private static List<Point> rectangle(int x1, int y1, int x2, int y2) {
        return List.of(new Point(x1, y1), new Point(x2, y1), new Point(x2, y2), new Point(x1, y2));
    }

    /** Every pixel distinct, so a misplaced copy shows. */
    private static BufferedImage pattern(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.setRGB(x, y, (x * 6) << 16 | (y * 7) << 8 | ((x + y) * 3));
            }
        }
        return image;
    }
}
