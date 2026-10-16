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

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.data.Offset.offset;

/**
 * Expected regions are the reference CRAFT implementation's output ({@code clovaai/CRAFT-pytorch},
 * {@code getDetBoxes_core} then {@code adjustResultCoordinates}) on the same score maps.
 */
class CraftPostProcessorTest {

    private static final int W = 80;
    private static final int H = 40;

    private final CraftPostProcessor defaults = new CraftPostProcessor(0.7f, 0.4f, 0.4f, 10);

    @Test
    void joinsCharactersLinkedIntoAWord() {
        float[] text = new float[W * H];
        float[] link = new float[W * H];
        fill(text, 10, 20, 10, 20, 0.9f);
        fill(text, 24, 34, 10, 20, 0.9f);
        fill(link, 18, 26, 12, 18, 0.8f);

        List<TextRegion> regions = defaults.process(text, link, W, H, 2.0, 160, 80);

        assertThat(regions).singleElement().satisfies(region -> {
            assertCorners(region, 14, 14, 72, 14, 72, 44, 14, 44);
            assertThat(region.confidence()).isEqualTo(0.9f);
            assertThat(region.box()).isEqualTo(new BoundingBox(14, 14, 72, 44));
        });
    }

    @Test
    void keepsAnAreaWhoseHighestTextScoreReachesTheThresholdEvenIfItsMeanDoesNot() {
        float[] text = new float[W * H];
        float[] link = new float[W * H];
        fill(text, 5, 35, 5, 15, 0.45f);    // mean 0.45
        text[10 * W + 20] = 0.95f;          // but one character center is clear
        fill(text, 45, 75, 25, 35, 0.65f);  // uniformly faint: dropped

        List<TextRegion> regions = defaults.process(text, link, W, H, 2.0, 160, 80);

        assertThat(regions).singleElement().satisfies(region -> {
            assertCorners(region, 4, 4, 74, 4, 74, 34, 4, 34);
            assertThat(region.confidence()).isEqualTo(0.95f);
        });
    }

    @Test
    void boxesRotatedTextWithARotatedRectangle() {
        int[][] rows = {{9, 16, 17}, {10, 16, 21}, {11, 15, 25}, {12, 15, 29}, {13, 15, 32}, {14, 15, 36},
                {15, 14, 40}, {16, 14, 44}, {17, 14, 48}, {18, 16, 51}, {19, 20, 55}, {20, 24, 59}, {21, 28, 63},
                {22, 31, 65}, {23, 35, 65}, {24, 39, 65}, {25, 43, 64}, {26, 47, 64}, {27, 50, 64}, {28, 54, 64},
                {29, 58, 63}, {30, 62, 63}};
        float[] text = new float[W * H];
        for (int[] row : rows) {
            for (int x = row[1]; x <= row[2]; x++) {
                text[row[0] * W + x] = 0.85f;
            }
        }

        List<TextRegion> regions = defaults.process(text, new float[W * H], W, H, 2.0, 160, 80);

        assertThat(regions).singleElement().satisfies(region -> assertCorners(region,
                26.0984f, 8.0259f, 140.2953f, 38.0777f, 131.9016f, 69.9741f, 17.7047f, 39.9223f));
    }

    @Test
    void boxesANearlySquareAreaAxisAlignedAndScalesToTheImage() {
        float[] text = new float[W * H];
        fill(text, 30, 42, 10, 22, 0.8f);

        List<TextRegion> regions = defaults.process(text, new float[W * H], W, H, 3.0, 240, 120);

        assertThat(regions).singleElement().satisfies(region -> assertCorners(region, 81, 21, 132, 21, 132, 72, 81, 72));
    }

    @Test
    void ignoresTinyAreasAndLinksWithoutCharacters() {
        float[] text = new float[W * H];
        float[] link = new float[W * H];
        fill(text, 2, 5, 2, 5, 0.95f);      // 9 pixels
        fill(link, 20, 60, 20, 30, 0.9f);   // link score only

        assertThat(defaults.process(text, link, W, H, 2.0, 160, 80)).isEmpty();
    }

    @Test
    void listsRegionsByConfidence() {
        float[] text = new float[W * H];
        fill(text, 5, 20, 5, 12, 0.75f);
        fill(text, 40, 60, 25, 32, 0.95f);

        List<TextRegion> regions = defaults.process(text, new float[W * H], W, H, 2.0, 160, 80);

        assertThat(regions).extracting(TextRegion::confidence).containsExactly(0.95f, 0.75f);
    }

    @Test
    void rejectsMapsOfTheWrongSize() {
        assertThatThrownBy(() -> defaults.process(new float[10], new float[10], 4, 4, 2.0, 8, 8))
                .isInstanceOf(IllegalArgumentException.class);
    }

    private static void fill(float[] map, int x0, int x1, int y0, int y1, float value) {
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                map[y * W + x] = value;
            }
        }
    }

    private static void assertCorners(TextRegion region, float... xy) {
        assertThat(region.polygon()).hasSize(4);
        for (int i = 0; i < 4; i++) {
            assertThat(region.polygon().get(i).x()).as("x of corner " + i).isCloseTo(xy[2 * i], offset(1e-3f));
            assertThat(region.polygon().get(i).y()).as("y of corner " + i).isCloseTo(xy[2 * i + 1], offset(1e-3f));
        }
    }
}
