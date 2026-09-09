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

/**
 * Maps a normalized value in {@code [0, 1]} to an RGB colour.
 *
 * <p>Dense outputs such as depth maps carry structure that grayscale flattens —
 * a perceptual colormap makes near and far readable at a glance. Each colormap is
 * defined by a small set of anchor colours interpolated linearly, so no lookup
 * table or external dependency is needed.
 *
 * @see TensorImages#toColormap(float[][], Colormap)
 */
public enum Colormap {

    /** Black to white. The plain choice, and the right one when the image is data rather than a picture. */
    GRAYSCALE(new int[][]{
            {0, 0, 0},
            {255, 255, 255}
    }),

    /** Perceptually uniform dark blue to yellow. Colour-blind friendly. */
    VIRIDIS(new int[][]{
            {68, 1, 84},
            {72, 40, 120},
            {62, 74, 137},
            {49, 104, 142},
            {38, 130, 142},
            {31, 158, 137},
            {53, 183, 121},
            {109, 205, 89},
            {180, 222, 44},
            {253, 231, 37}
    }),

    /** High-contrast rainbow, dark blue through green to deep red. Good for depth. */
    TURBO(new int[][]{
            {48, 18, 59},
            {70, 107, 227},
            {54, 168, 251},
            {24, 214, 203},
            {72, 240, 130},
            {163, 253, 60},
            {227, 220, 55},
            {253, 149, 39},
            {220, 65, 12},
            {122, 4, 3}
    });

    private final int[][] anchors;

    Colormap(int[][] anchors) {
        this.anchors = anchors;
    }

    /**
     * Returns the packed {@code 0xRRGGBB} colour for a normalized value.
     *
     * <p>Values outside {@code [0, 1]} are clamped to the ends of the ramp.
     *
     * @param value a normalized value, typically in {@code [0, 1]}
     * @return the packed RGB colour
     */
    public int rgb(float value) {
        float clamped = Math.max(0f, Math.min(1f, value));
        int segments = anchors.length - 1;
        float scaled = clamped * segments;
        int index = (int) scaled;
        if (index >= segments) {
            index = segments - 1;
        }
        float t = scaled - index;

        int[] from = anchors[index];
        int[] to = anchors[index + 1];
        int r = Math.round(from[0] + (to[0] - from[0]) * t);
        int g = Math.round(from[1] + (to[1] - from[1]) * t);
        int b = Math.round(from[2] + (to[2] - from[2]) * t);
        return (r << 16) | (g << 8) | b;
    }
}
