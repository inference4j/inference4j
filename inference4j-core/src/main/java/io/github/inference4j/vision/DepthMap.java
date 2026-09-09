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

/**
 * A per-pixel depth prediction for an image.
 *
 * <p>Values are <strong>relative inverse depth</strong>, not metres: larger means
 * closer to the camera. The scale is model-dependent and differs between images,
 * so compare values within one map, never across two.
 *
 * <p>Note that {@code values} is a mutable array, so record equality compares array
 * identity rather than contents. Depth maps are not meaningfully comparable anyway,
 * but it is worth stating rather than discovering.
 *
 * @param values per-pixel depth as {@code [height][width]}, in original image coordinates
 * @param width  map width in pixels
 * @param height map height in pixels
 * @see DepthEstimator
 */
public record DepthMap(float[][] values, int width, int height) {

    /**
     * Returns the depth at a pixel.
     *
     * @param x column, from the left edge
     * @param y row, from the top edge
     * @return relative inverse depth at that pixel
     * @throws IndexOutOfBoundsException if the coordinates fall outside the map
     */
    public float at(int x, int y) {
        return values[y][x];
    }

    /** Returns the smallest depth value in the map — the furthest point. */
    public float min() {
        float min = Float.POSITIVE_INFINITY;
        for (float[] row : values) {
            for (float v : row) {
                if (v < min) {
                    min = v;
                }
            }
        }
        return min;
    }

    /** Returns the largest depth value in the map — the nearest point. */
    public float max() {
        float max = Float.NEGATIVE_INFINITY;
        for (float[] row : values) {
            for (float v : row) {
                if (v > max) {
                    max = v;
                }
            }
        }
        return max;
    }

    /**
     * Renders the map as a grayscale image, rescaled to its own range.
     *
     * @return a grayscale image, brighter where closer
     */
    public BufferedImage toImage() {
        return TensorImages.toGrayscale(values);
    }

    /**
     * Renders the map as an image using a colormap, rescaled to its own range.
     *
     * @param colormap the colormap to apply
     * @return a colour image
     */
    public BufferedImage toImage(Colormap colormap) {
        return TensorImages.toColormap(values, colormap);
    }
}
