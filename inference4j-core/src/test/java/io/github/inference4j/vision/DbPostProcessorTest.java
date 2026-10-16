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
 * Expected polygons and scores are PaddleOCR's {@code DBPostProcess} output (paddlex 3.7, {@code thresh 0.3},
 * {@code box_thresh 0.6}, {@code unclip_ratio 1.5}) on the same maps.
 */
class DbPostProcessorTest {

    private final DbPostProcessor processor = new DbPostProcessor(0.3f, 0.6f, 1.5f, 3000);

    @Test
    void expandsAnAxisAlignedLineLikePaddleOcr() {
        float[] map = bar(64, 32, 8, 56, 10, 20, 0.9f);

        List<TextRegion> regions = processor.process(map, 64, 32, 64, 32);

        assertThat(regions).hasSize(1);
        assertThat(regions.get(0).polygon()).containsExactly(
                new Point(2, 3), new Point(60, 3), new Point(60, 24), new Point(2, 24));
        assertThat(regions.get(0).confidence()).isCloseTo(0.851678f, offset(1e-5f));
        assertThat(regions.get(0).box()).isEqualTo(new BoundingBox(2, 3, 60, 24));
    }

    @Test
    void scalesRegionsToTheOriginalImage() {
        float[] map = bar(64, 32, 8, 56, 10, 20, 0.9f);

        List<TextRegion> regions = processor.process(map, 64, 32, 128, 64);

        assertThat(regions.get(0).polygon()).containsExactly(
                new Point(4, 6), new Point(120, 6), new Point(120, 48), new Point(4, 48));
    }

    @Test
    void keepsRotatedLinesRotated() {
        int[][] rows = {{19, 24, 25}, {20, 24, 28}, {21, 23, 30}, {22, 23, 33}, {23, 23, 36}, {24, 22, 38},
                {25, 22, 41}, {26, 22, 44}, {27, 22, 46}, {28, 21, 49}, {29, 21, 52}, {30, 21, 54}, {31, 20, 57},
                {32, 20, 60}, {33, 22, 62}, {34, 25, 65}, {35, 27, 68}, {36, 30, 70}, {37, 33, 73}, {38, 35, 76},
                {39, 38, 78}, {40, 41, 81}, {41, 43, 84}, {42, 46, 86}, {43, 49, 89}, {44, 51, 92}, {45, 54, 94},
                {46, 57, 97}, {47, 59, 99}, {48, 62, 99}, {49, 65, 98}, {50, 67, 98}, {51, 70, 98}, {52, 73, 97},
                {53, 75, 97}, {54, 78, 97}, {55, 81, 97}, {56, 83, 96}, {57, 86, 96}, {58, 89, 96}, {59, 91, 95},
                {60, 94, 95}};
        float[] map = new float[120 * 80];
        for (int[] row : rows) {
            for (int x = row[1]; x <= row[2]; x++) {
                map[row[0] * 120 + x] = 0.8f;
            }
        }

        List<TextRegion> regions = processor.process(map, 120, 80, 120, 80);

        assertThat(regions).hasSize(1);
        assertThat(regions.get(0).polygon()).containsExactly(
                new Point(17, 6), new Point(112, 41), new Point(101, 72), new Point(6, 37));
        assertThat(regions.get(0).confidence()).isCloseTo(0.754729f, offset(1e-5f));
        assertThat(regions.get(0).box()).isEqualTo(new BoundingBox(6, 6, 112, 72));
    }

    @Test
    void dropsTinyAreasAndFaintLines() {
        float[] map = bar(64, 32, 10, 50, 15, 25, 0.4f);   // above threshold, below boxThreshold
        map[5 * 64 + 5] = map[5 * 64 + 6] = map[6 * 64 + 5] = map[6 * 64 + 6] = 0.9f;   // a 2 × 2 speck

        assertThat(processor.process(map, 64, 32, 64, 32)).isEmpty();
    }

    @Test
    void rejectsAMapOfTheWrongSize() {
        assertThatThrownBy(() -> processor.process(new float[10], 4, 4, 4, 4))
                .isInstanceOf(IllegalArgumentException.class);
    }

    private static float[] bar(int width, int height, int x0, int x1, int y0, int y1, float value) {
        float[] map = new float[width * height];
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                map[y * width + x] = value;
            }
        }
        return map;
    }
}
