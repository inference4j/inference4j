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

/**
 * Converts dense model output back into viewable images.
 *
 * <p>Models that predict a value per pixel — depth estimation, segmentation,
 * super-resolution — produce float arrays rather than images. These helpers turn
 * those arrays into a {@link BufferedImage} you can display or write to disk with
 * {@link ImageAnnotator#save(BufferedImage, java.nio.file.Path)}.
 *
 * <p>Raw model values are rarely in display range, so {@link #toGrayscale(float[][])}
 * and {@link #toColormap(float[][], Colormap)} rescale to the array's own minimum and
 * maximum. {@link #toRgb(float[][][])} does not — it assumes values already represent
 * 0&ndash;255 intensities and clamps.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * float[][] depth = outputs.get("predicted_depth").squeeze(0).toFloats2D();
 * BufferedImage image = TensorImages.toColormap(depth, Colormap.TURBO);
 * ImageAnnotator.save(image, Path.of("depth.png"));
 * }</pre>
 *
 * @see Colormap
 * @see ImageAnnotator
 */
public final class TensorImages {

    private TensorImages() {
    }

    /**
     * Converts a 2D value map to a grayscale image, rescaled to its own range.
     *
     * @param values a {@code [height][width]} array
     * @return a grayscale image the same size as the input
     * @throws InferenceException if the array is empty or rows have differing lengths
     */
    public static BufferedImage toGrayscale(float[][] values) {
        return toColormap(values, Colormap.GRAYSCALE);
    }

    /**
     * Converts a 2D value map to an image using a colormap, rescaled to its own range.
     *
     * <p>A perceptual colormap makes structure visible that grayscale flattens —
     * worth preferring for depth maps and heatmaps.
     *
     * @param values   a {@code [height][width]} array
     * @param colormap the colormap to apply
     * @return an RGB image the same size as the input
     * @throws InferenceException if the array is empty or rows have differing lengths
     */
    public static BufferedImage toColormap(float[][] values, Colormap colormap) {
        int height = values.length;
        if (height == 0) {
            throw new InferenceException("Cannot convert an empty array to an image");
        }
        int width = values[0].length;
        if (width == 0) {
            throw new InferenceException("Cannot convert an empty array to an image");
        }

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float[] row : values) {
            if (row.length != width) {
                throw new InferenceException(
                        "Ragged array: expected every row to have width " + width
                                + " but found " + row.length);
            }
            for (float v : row) {
                if (v < min) {
                    min = v;
                }
                if (v > max) {
                    max = v;
                }
            }
        }

        float range = max - min;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float normalized = range > 0f ? (values[y][x] - min) / range : 0f;
                image.setRGB(x, y, colormap.rgb(normalized));
            }
        }
        return image;
    }

    /**
     * Converts a 3-channel CHW array to an RGB image.
     *
     * <p>Values are treated as 0&ndash;255 intensities and clamped, so denormalize
     * model output before calling this. Model outputs are typically
     * {@code [1, 3, height, width]}, so squeeze the batch dimension first:
     * {@code tensor.squeeze(0).toFloats3D()}.
     *
     * @param chw a {@code [3][height][width]} array
     * @return an RGB image
     * @throws InferenceException if the array is not 3-channel, is empty, or is ragged
     */
    public static BufferedImage toRgb(float[][][] chw) {
        if (chw.length != 3) {
            throw new InferenceException(
                    "Expected 3 channels for RGB conversion but found " + chw.length);
        }
        int height = chw[0].length;
        if (height == 0 || chw[0][0].length == 0) {
            throw new InferenceException("Cannot convert an empty array to an image");
        }
        int width = chw[0][0].length;

        for (int c = 0; c < 3; c++) {
            if (chw[c].length != height) {
                throw new InferenceException(
                        "Channel " + c + " has height " + chw[c].length + ", expected " + height);
            }
            for (float[] row : chw[c]) {
                if (row.length != width) {
                    throw new InferenceException(
                            "Ragged array: expected every row to have width " + width
                                    + " but found " + row.length);
                }
            }
        }

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int r = clampToByte(chw[0][y][x]);
                int g = clampToByte(chw[1][y][x]);
                int b = clampToByte(chw[2][y][x]);
                image.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return image;
    }

    private static int clampToByte(float value) {
        int rounded = Math.round(value);
        return Math.max(0, Math.min(255, rounded));
    }
}
