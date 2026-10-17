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
 * Image resizing as Pillow's {@code Image.resize(..., BILINEAR)} does it, which HuggingFace image processors use:
 * a triangle filter whose support widens with the reduction (so shrinking averages, rather than samples, the
 * source pixels), applied horizontally then vertically in 8-bit fixed point.
 */
final class PilImages {

    private static final int PRECISION_BITS = 32 - 8 - 2;

    private PilImages() {
    }

    /** Resizes packed RGB pixels, bit for bit as Pillow does for an RGB image. */
    static int[] resizeBilinear(int[] src, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
        int[] horizontal = srcWidth == dstWidth ? src
                : resizeAxis(src, srcWidth, srcHeight, dstWidth, true);
        return srcHeight == dstHeight ? horizontal.clone()
                : resizeAxis(horizontal, dstWidth, srcHeight, dstHeight, false);
    }

    /** One pass of Pillow's {@code ImagingResample{Horizontal,Vertical}_8bpc}. */
    private static int[] resizeAxis(int[] src, int width, int height, int outSize, boolean horizontal) {
        int inSize = horizontal ? width : height;
        Coefficients c = coefficients(inSize, outSize);
        int outWidth = horizontal ? outSize : width;
        int outHeight = horizontal ? height : outSize;
        int[] dst = new int[outWidth * outHeight];
        int half = 1 << (PRECISION_BITS - 1);
        for (int y = 0; y < outHeight; y++) {
            for (int x = 0; x < outWidth; x++) {
                int o = horizontal ? x : y;
                int min = c.bounds[2 * o];
                int count = c.bounds[2 * o + 1];
                int r = half, g = half, b = half;
                for (int i = 0; i < count; i++) {
                    int rgb = horizontal ? src[y * width + min + i] : src[(min + i) * width + x];
                    int k = c.kernel[o * c.size + i];
                    r += ((rgb >> 16) & 0xFF) * k;
                    g += ((rgb >> 8) & 0xFF) * k;
                    b += (rgb & 0xFF) * k;
                }
                dst[y * outWidth + x] = (clip8(r) << 16) | (clip8(g) << 8) | clip8(b);
            }
        }
        return dst;
    }

    private record Coefficients(int size, int[] bounds, int[] kernel) {
    }

    /** Pillow's {@code precompute_coeffs} with the bilinear (triangle) filter, then {@code normalize_coeffs_8bpc}. */
    private static Coefficients coefficients(int inSize, int outSize) {
        double scale = (double) inSize / outSize;
        double filterScale = Math.max(scale, 1.0);
        double support = filterScale;   // bilinear support is 1.0
        int size = (int) Math.ceil(support) * 2 + 1;
        int[] bounds = new int[outSize * 2];
        int[] kernel = new int[outSize * size];
        double[] weights = new double[size];
        for (int xx = 0; xx < outSize; xx++) {
            double center = (xx + 0.5) * scale;
            double ss = 1.0 / filterScale;
            int min = Math.max(0, (int) (center - support + 0.5));
            int max = Math.min(inSize, (int) (center + support + 0.5)) - min;
            double total = 0.0;
            for (int x = 0; x < max; x++) {
                double w = triangle((x + min - center + 0.5) * ss);
                weights[x] = w;
                total += w;
            }
            for (int x = 0; x < max; x++) {
                double w = total != 0.0 ? weights[x] / total : weights[x];
                kernel[xx * size + x] = w < 0
                        ? (int) (-0.5 + w * (1 << PRECISION_BITS))
                        : (int) (0.5 + w * (1 << PRECISION_BITS));
            }
            bounds[2 * xx] = min;
            bounds[2 * xx + 1] = max;
        }
        return new Coefficients(size, bounds, kernel);
    }

    private static double triangle(double x) {
        x = Math.abs(x);
        return x < 1.0 ? 1.0 - x : 0.0;
    }

    private static int clip8(int value) {
        int shifted = value >> PRECISION_BITS;
        return shifted < 0 ? 0 : Math.min(shifted, 255);
    }
}
