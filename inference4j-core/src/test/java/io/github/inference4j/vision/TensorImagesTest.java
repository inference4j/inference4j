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

import io.github.inference4j.exception.InferenceException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class TensorImagesTest {

    // --- Grayscale ---

    @Test
    void toGrayscale_mapsRangeToFullBlackAndWhite() {
        float[][] values = {{0f, 5f}, {10f, 2.5f}};
        BufferedImage image = TensorImages.toGrayscale(values);

        assertThat(image.getWidth()).isEqualTo(2);
        assertThat(image.getHeight()).isEqualTo(2);
        assertThat(image.getRGB(0, 0)).isEqualTo(0xFF000000);
        assertThat(image.getRGB(0, 1)).isEqualTo(0xFFFFFFFF);
    }

    @Test
    void toGrayscale_rescalesToOwnRangeNotAbsoluteValues() {
        // 100..200 should span black to white just as 0..1 does
        BufferedImage image = TensorImages.toGrayscale(new float[][]{{100f, 200f}});
        assertThat(image.getRGB(0, 0)).isEqualTo(0xFF000000);
        assertThat(image.getRGB(1, 0)).isEqualTo(0xFFFFFFFF);
    }

    @Test
    void toGrayscale_constantMapDoesNotDivideByZero() {
        BufferedImage image = TensorImages.toGrayscale(new float[][]{{3f, 3f}, {3f, 3f}});
        assertThat(image.getRGB(0, 0)).isEqualTo(0xFF000000);
        assertThat(image.getRGB(1, 1)).isEqualTo(0xFF000000);
    }

    @Test
    void toGrayscale_preservesRowColumnOrientation() {
        // values[y][x] — a value in row 1, column 0 must land at pixel (0, 1)
        float[][] values = {{0f, 0f}, {10f, 0f}};
        BufferedImage image = TensorImages.toGrayscale(values);
        assertThat(image.getRGB(0, 1)).isEqualTo(0xFFFFFFFF);
        assertThat(image.getRGB(1, 0)).isEqualTo(0xFF000000);
    }

    @Test
    void toGrayscale_throwsOnEmpty() {
        assertThatThrownBy(() -> TensorImages.toGrayscale(new float[0][0]))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("empty");
    }

    @Test
    void toGrayscale_throwsOnRaggedRows() {
        float[][] ragged = {{1f, 2f}, {3f}};
        assertThatThrownBy(() -> TensorImages.toGrayscale(ragged))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("Ragged");
    }

    // --- Colormap ---

    @Test
    void toColormap_appliesRampEnds() {
        BufferedImage image = TensorImages.toColormap(new float[][]{{0f, 1f}}, Colormap.TURBO);
        assertThat(image.getRGB(0, 0)).isEqualTo(0xFF000000 | Colormap.TURBO.rgb(0f));
        assertThat(image.getRGB(1, 0)).isEqualTo(0xFF000000 | Colormap.TURBO.rgb(1f));
    }

    @Test
    void toColormap_producesColourNotGray() {
        BufferedImage image = TensorImages.toColormap(new float[][]{{0f, 0.5f, 1f}}, Colormap.VIRIDIS);
        int mid = image.getRGB(1, 0);
        int r = (mid >> 16) & 0xFF;
        int g = (mid >> 8) & 0xFF;
        int b = mid & 0xFF;
        assertThat(r == g && g == b).as("Viridis midpoint should not be gray").isFalse();
    }

    // --- RGB ---

    @Test
    void toRgb_buildsImageFromChannelPlanes() {
        float[][][] chw = {
                {{255f, 0f}},
                {{0f, 255f}},
                {{0f, 0f}}
        };
        BufferedImage image = TensorImages.toRgb(chw);

        assertThat(image.getWidth()).isEqualTo(2);
        assertThat(image.getHeight()).isEqualTo(1);
        assertThat(image.getRGB(0, 0) & 0xFFFFFF).isEqualTo(0xFF0000);
        assertThat(image.getRGB(1, 0) & 0xFFFFFF).isEqualTo(0x00FF00);
    }

    @Test
    void toRgb_clampsOutOfRangeValues() {
        float[][][] chw = {{{-50f}}, {{300f}}, {{128f}}};
        BufferedImage image = TensorImages.toRgb(chw);
        assertThat(image.getRGB(0, 0) & 0xFFFFFF).isEqualTo(0x00FF80);
    }

    @Test
    void toRgb_throwsOnWrongChannelCount() {
        float[][][] chw = {{{1f}}, {{2f}}};
        assertThatThrownBy(() -> TensorImages.toRgb(chw))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("3 channels");
    }

    // --- Colormap ramp ---

    @Test
    void colormap_clampsOutsideUnitRange() {
        assertThat(Colormap.VIRIDIS.rgb(-1f)).isEqualTo(Colormap.VIRIDIS.rgb(0f));
        assertThat(Colormap.VIRIDIS.rgb(2f)).isEqualTo(Colormap.VIRIDIS.rgb(1f));
    }

    @Test
    void colormap_grayscaleIsMonotonicAndNeutral() {
        int mid = Colormap.GRAYSCALE.rgb(0.5f);
        int r = (mid >> 16) & 0xFF;
        int g = (mid >> 8) & 0xFF;
        int b = mid & 0xFF;
        assertThat(r).isEqualTo(g).isEqualTo(b);
        assertThat(Colormap.GRAYSCALE.rgb(0f)).isEqualTo(0x000000);
        assertThat(Colormap.GRAYSCALE.rgb(1f)).isEqualTo(0xFFFFFF);
    }

    @Test
    void colormap_interpolatesBetweenAnchors() {
        // A value between anchors must not simply snap to one of them
        int low = Colormap.TURBO.rgb(0f);
        int high = Colormap.TURBO.rgb(1f);
        int mid = Colormap.TURBO.rgb(0.5f);
        assertThat(mid).isNotEqualTo(low).isNotEqualTo(high);
    }
}
