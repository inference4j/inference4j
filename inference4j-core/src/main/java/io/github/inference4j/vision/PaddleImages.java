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
import java.nio.file.Path;

/**
 * Image handling shared by the PaddleOCR wrappers, matching OpenCV where PaddleOCR relies on it: pixels as
 * packed RGB, and bilinear and bicubic resizing as {@code cv2.resize} does them.
 */
final class PaddleImages {

    private PaddleImages() {
    }

    static int[] pixels(BufferedImage image) {
        return DetectionOps.pixels(image);
    }

    /**
     * Bilinear resize, bit for bit as OpenCV 4.x's {@code cv2.resize(..., INTER_LINEAR)} on 8-bit images: pixel
     * centers aligned, 11-bit fixed-point weights, and OpenCV's integer rounding. An exact 2× reduction averages
     * 2×2 blocks, as OpenCV switches to area interpolation for it. Matching OpenCV exactly matters: the recognizer's
     * confidences react to ±1 differences in many pixels.
     */
    static int[] resize(int[] src, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
        if (srcWidth == dstWidth && srcHeight == dstHeight) {
            return src.clone();
        }
        double scaleX = 1.0 / ((double) dstWidth / srcWidth);
        double scaleY = 1.0 / ((double) dstHeight / srcHeight);
        if (scaleX == 2.0 && scaleY == 2.0) {
            return halve(src, srcWidth, dstWidth, dstHeight);
        }

        int[] xOffsets = new int[dstWidth];
        int[] alpha0 = new int[dstWidth];
        int[] alpha1 = new int[dstWidth];
        boolean[] edge = new boolean[dstWidth];   // the right neighbor is outside: the pixel alone counts
        for (int dx = 0; dx < dstWidth; dx++) {
            float fx = (float) ((dx + 0.5) * scaleX - 0.5);
            int sx = (int) Math.floor(fx);
            fx -= sx;
            if (sx < 0) {
                fx = 0;
                sx = 0;
            }
            if (sx + 1 >= srcWidth) {
                edge[dx] = true;
                if (sx >= srcWidth - 1) {
                    fx = 0;
                    sx = srcWidth - 1;
                }
            }
            xOffsets[dx] = sx;
            alpha0[dx] = (int) Math.rint((1f - fx) * COEF_SCALE);
            alpha1[dx] = (int) Math.rint(fx * COEF_SCALE);
        }

        int[] dst = new int[dstWidth * dstHeight];
        int[][] rows = new int[2][3 * dstWidth];
        for (int dy = 0; dy < dstHeight; dy++) {
            float fy = (float) ((dy + 0.5) * scaleY - 0.5);
            int sy = (int) Math.floor(fy);
            fy -= sy;
            int beta0 = (int) Math.rint((1f - fy) * COEF_SCALE);
            int beta1 = (int) Math.rint(fy * COEF_SCALE);
            for (int k = 0; k < 2; k++) {
                int row = Math.max(0, Math.min(sy + k, srcHeight - 1));
                horizontal(src, row * srcWidth, srcWidth, xOffsets, alpha0, alpha1, edge, rows[k]);
            }
            for (int dx = 0; dx < dstWidth; dx++) {
                int pixel = 0;
                for (int c = 0; c < 3; c++) {
                    int s0 = rows[0][3 * dx + c];
                    int s1 = rows[1][3 * dx + c];
                    int value = (((beta0 * (s0 >> 4)) >> 16) + ((beta1 * (s1 >> 4)) >> 16) + 2) >> 2;
                    pixel |= Math.max(0, Math.min(255, value)) << (16 - 8 * c);
                }
                dst[dy * dstWidth + dx] = pixel;
            }
        }
        return dst;
    }

    private static final int COEF_SCALE = 2048;   // OpenCV's INTER_RESIZE_COEF_SCALE

    /**
     * Bicubic resize, as OpenCV 4.x's {@code cv2.resize(..., INTER_CUBIC)} on 8-bit images: pixel centers aligned,
     * the Keys kernel with a = -0.75 in 11-bit fixed point, edge pixels repeated beyond the border. Unlike Pillow,
     * OpenCV does not widen the kernel when shrinking.
     */
    static int[] resizeCubic(int[] src, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
        if (srcWidth == dstWidth && srcHeight == dstHeight) {
            return src.clone();
        }
        double scaleX = 1.0 / ((double) dstWidth / srcWidth);
        double scaleY = 1.0 / ((double) dstHeight / srcHeight);

        int[] xOffsets = new int[dstWidth];
        int[][] alpha = new int[dstWidth][];
        for (int dx = 0; dx < dstWidth; dx++) {
            float fx = (float) ((dx + 0.5) * scaleX - 0.5);
            int sx = (int) Math.floor(fx);
            xOffsets[dx] = sx;
            alpha[dx] = cubicCoefficients(fx - sx);
        }

        // Horizontally resized rows, computed once per source row
        int[][] rows = new int[srcHeight][];
        int[] dst = new int[dstWidth * dstHeight];
        for (int dy = 0; dy < dstHeight; dy++) {
            float fy = (float) ((dy + 0.5) * scaleY - 0.5);
            int sy = (int) Math.floor(fy);
            int[] beta = cubicCoefficients(fy - sy);
            float[] betaF = new float[4];
            for (int k = 0; k < 4; k++) {
                betaF[k] = beta[k] * (1f / (COEF_SCALE * COEF_SCALE));
            }
            int[][] taps = new int[4][];
            for (int k = 0; k < 4; k++) {
                int row = Math.max(0, Math.min(sy - 1 + k, srcHeight - 1));
                if (rows[row] == null) {
                    rows[row] = horizontalCubic(src, row * srcWidth, srcWidth, xOffsets, alpha);
                }
                taps[k] = rows[row];
            }
            for (int dx = 0; dx < dstWidth; dx++) {
                int pixel = 0;
                for (int c = 0; c < 3; c++) {
                    int i = 3 * dx + c;
                    // OpenCV's vectorized VResizeCubicVec_32s8u: float weights, fused multiply-adds, round to even
                    float sum = taps[3][i] * betaF[3];
                    sum = Math.fma((float) taps[2][i], betaF[2], sum);
                    sum = Math.fma((float) taps[1][i], betaF[1], sum);
                    sum = Math.fma((float) taps[0][i], betaF[0], sum);
                    int value = (int) Math.rint(sum);
                    pixel |= Math.max(0, Math.min(255, value)) << (16 - 8 * c);
                }
                dst[dy * dstWidth + dx] = pixel;
            }
        }
        return dst;
    }

    /** OpenCV's {@code interpolateCubic} in float, then {@code saturate_cast<short>} of the 11-bit scaled weights. */
    private static int[] cubicCoefficients(float x) {
        final float a = -0.75f;
        float c0 = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a;
        float c1 = ((a + 2) * x - (a + 3)) * x * x + 1;
        float c2 = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1;
        float c3 = 1f - c0 - c1 - c2;
        return new int[]{(int) Math.rint(c0 * COEF_SCALE), (int) Math.rint(c1 * COEF_SCALE),
                (int) Math.rint(c2 * COEF_SCALE), (int) Math.rint(c3 * COEF_SCALE)};
    }

    /** One source row resized horizontally with the cubic weights: channels R, G, B per output pixel. */
    private static int[] horizontalCubic(int[] src, int rowStart, int srcWidth, int[] xOffsets, int[][] alpha) {
        int[] out = new int[3 * xOffsets.length];
        for (int dx = 0; dx < xOffsets.length; dx++) {
            for (int j = 0; j < 4; j++) {
                int x = Math.max(0, Math.min(xOffsets[dx] - 1 + j, srcWidth - 1));
                int p = src[rowStart + x];
                int weight = alpha[dx][j];
                out[3 * dx] += weight * ((p >> 16) & 0xFF);
                out[3 * dx + 1] += weight * ((p >> 8) & 0xFF);
                out[3 * dx + 2] += weight * (p & 0xFF);
            }
        }
        return out;
    }

    /** One source row resized horizontally, in OpenCV's fixed point: channels R, G, B per output pixel. */
    private static void horizontal(int[] src, int rowStart, int srcWidth, int[] xOffsets, int[] alpha0, int[] alpha1,
                                   boolean[] edge, int[] out) {
        for (int dx = 0; dx < xOffsets.length; dx++) {
            int p0 = src[rowStart + xOffsets[dx]];
            int p1 = edge[dx] ? p0 : src[rowStart + xOffsets[dx] + 1];
            for (int c = 0; c < 3; c++) {
                int shift = 16 - 8 * c;
                int v0 = (p0 >> shift) & 0xFF;
                int v1 = (p1 >> shift) & 0xFF;
                out[3 * dx + c] = edge[dx] ? v0 * COEF_SCALE : v0 * alpha0[dx] + v1 * alpha1[dx];
            }
        }
    }

    /** OpenCV's fast area path for an exact 2× reduction: the rounded mean of each 2×2 block. */
    private static int[] halve(int[] src, int srcWidth, int dstWidth, int dstHeight) {
        int[] dst = new int[dstWidth * dstHeight];
        for (int y = 0; y < dstHeight; y++) {
            for (int x = 0; x < dstWidth; x++) {
                int a = src[2 * y * srcWidth + 2 * x];
                int b = src[2 * y * srcWidth + 2 * x + 1];
                int c = src[(2 * y + 1) * srcWidth + 2 * x];
                int d = src[(2 * y + 1) * srcWidth + 2 * x + 1];
                int pixel = 0;
                for (int shift = 16; shift >= 0; shift -= 8) {
                    int sum = ((a >> shift) & 0xFF) + ((b >> shift) & 0xFF) + ((c >> shift) & 0xFF) + ((d >> shift) & 0xFF);
                    pixel |= ((sum + 2) >> 2) << shift;
                }
                dst[y * dstWidth + x] = pixel;
            }
        }
        return dst;
    }

    static BufferedImage read(Path path) {
        return ImageFiles.read(path);
    }
}
