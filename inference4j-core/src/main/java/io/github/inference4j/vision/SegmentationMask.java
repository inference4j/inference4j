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
import java.util.Optional;

/**
 * A per-pixel mask of one object or region in an image.
 *
 * <p>{@code true} marks pixels that belong to the object. The mask is always the same size as
 * the image it was predicted for, so it can be applied straight back to that image:
 * {@link #applyTo(BufferedImage)} cuts the object out on a transparent background and
 * {@link #overlay(BufferedImage, Color)} highlights it in place.
 *
 * <p>Note that {@code mask} is a mutable array, so record equality compares array identity
 * rather than contents.
 *
 * @param mask   per-pixel membership as {@code [height][width]}, in original image coordinates
 * @param width  mask width in pixels
 * @param height mask height in pixels
 * @param score  the model's confidence in this mask; for SAM, its predicted IoU with the true
 *               object, between 0 and 1
 * @see PromptableSegmenter
 */
public record SegmentationMask(boolean[][] mask, int width, int height, float score) {

    private static final float OVERLAY_ALPHA = 0.5f;

    /**
     * Returns whether a pixel belongs to the object.
     *
     * @param x column, from the left edge
     * @param y row, from the top edge
     * @return {@code true} if the pixel is part of the mask
     * @throws IndexOutOfBoundsException if the coordinates fall outside the mask
     */
    public boolean contains(int x, int y) {
        return mask[y][x];
    }

    /** Returns the number of pixels in the mask. */
    public int area() {
        int area = 0;
        for (boolean[] row : mask) {
            for (boolean inside : row) {
                if (inside) {
                    area++;
                }
            }
        }
        return area;
    }

    /** Returns whether no pixel belongs to the mask. */
    public boolean isEmpty() {
        return area() == 0;
    }

    /**
     * Returns the tightest box around the mask.
     *
     * <p>The box uses the same convention as {@link BoundingBox} elsewhere: {@code x2} and
     * {@code y2} are exclusive, so a single-pixel mask at (3, 4) has box (3, 4, 4, 5).
     *
     * @return the box, or empty if the mask has no pixels
     */
    public Optional<BoundingBox> boundingBox() {
        int minX = width;
        int minY = height;
        int maxX = -1;
        int maxY = -1;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x]) {
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                }
            }
        }
        if (maxX < 0) {
            return Optional.empty();
        }
        return Optional.of(new BoundingBox(minX, minY, maxX + 1, maxY + 1));
    }

    /**
     * Renders the mask as a black and white image: white inside, black outside.
     *
     * @return a grayscale image the size of the mask
     */
    public BufferedImage toImage() {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.setRGB(x, y, mask[y][x] ? 0xFFFFFFFF : 0xFF000000);
            }
        }
        return image;
    }

    /**
     * Cuts the object out of an image, making every pixel outside the mask transparent.
     *
     * <p>Save the result as PNG to keep the transparency; JPEG has no alpha channel.
     *
     * @param image the image the mask was predicted for
     * @return a new ARGB image the size of the mask
     * @throws IllegalArgumentException if the image size differs from the mask
     */
    public BufferedImage applyTo(BufferedImage image) {
        requireSameSize(image);
        BufferedImage cutout = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x]) {
                    cutout.setRGB(x, y, 0xFF000000 | (image.getRGB(x, y) & 0x00FFFFFF));
                }
            }
        }
        return cutout;
    }

    /**
     * Highlights the object by blending a colour over the masked pixels.
     *
     * @param image the image the mask was predicted for
     * @param color the highlight colour, blended at 50%
     * @return a new RGB image the size of the mask
     * @throws IllegalArgumentException if the image size differs from the mask
     */
    public BufferedImage overlay(BufferedImage image, Color color) {
        requireSameSize(image);
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                result.setRGB(x, y, mask[y][x] ? blend(rgb, color) : rgb);
            }
        }
        return result;
    }

    private static int blend(int rgb, Color color) {
        int r = blendChannel((rgb >> 16) & 0xFF, color.getRed());
        int g = blendChannel((rgb >> 8) & 0xFF, color.getGreen());
        int b = blendChannel(rgb & 0xFF, color.getBlue());
        return (r << 16) | (g << 8) | b;
    }

    private static int blendChannel(int base, int tint) {
        return Math.round(base * (1 - OVERLAY_ALPHA) + tint * OVERLAY_ALPHA);
    }

    private void requireSameSize(BufferedImage image) {
        if (image.getWidth() != width || image.getHeight() != height) {
            throw new IllegalArgumentException("Image is " + image.getWidth() + "x" + image.getHeight()
                    + " but the mask is " + width + "x" + height);
        }
    }
}
