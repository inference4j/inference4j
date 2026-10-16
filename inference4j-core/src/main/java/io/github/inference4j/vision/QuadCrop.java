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
import java.util.List;

/**
 * Cuts a text region out of an image as an upright rectangle, ready for a {@link TextRecognizer}.
 *
 * <p>Follows PaddleOCR's crop: the region's minimum-area rectangle is warped to an axis-aligned image with a
 * perspective transform (bicubic interpolation, edge pixels repeated beyond the border), and a crop at least
 * 1.5 times taller than wide, such as a vertical line, is turned 90° counter-clockwise.
 */
public final class QuadCrop {

    private static final double CUBIC_A = -0.75;   // OpenCV's bicubic kernel
    private static final int SUBPIXEL_STEPS = 32;   // OpenCV quantizes warp coordinates to 1/32 pixel

    private QuadCrop() {
    }

    /**
     * Crops the region outlined by {@code polygon} from {@code image}.
     *
     * @param image   the source image
     * @param polygon the region's outline (at least three points), in image pixels
     * @return the region as an upright image
     */
    public static BufferedImage crop(BufferedImage image, List<Point> polygon) {
        if (polygon.size() < 3) {
            throw new IllegalArgumentException("A crop polygon needs at least 3 points, got " + polygon.size());
        }
        double[] xs = new double[polygon.size()];
        double[] ys = new double[polygon.size()];
        for (int i = 0; i < polygon.size(); i++) {
            xs[i] = (int) polygon.get(i).x();
            ys[i] = (int) polygon.get(i).y();
        }
        Quads.RotatedRect rect = Quads.minAreaRectOfPolygon(xs, ys);
        double[][] corners = Quads.orderCorners(rect.xs(), rect.ys());

        int width = Math.max(1, (int) Math.max(distance(corners[0], corners[1]), distance(corners[2], corners[3])));
        int height = Math.max(1, (int) Math.max(distance(corners[0], corners[3]), distance(corners[1], corners[2])));
        double[] h = homography(width, height, corners);

        int srcWidth = image.getWidth();
        int srcHeight = image.getHeight();
        int[] src = image.getRGB(0, 0, srcWidth, srcHeight, null, 0, srcWidth);
        int[] dst = new int[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double w = h[6] * x + h[7] * y + h[8];
                double scale = w != 0 ? SUBPIXEL_STEPS / w : 0;
                long qx = Math.round((h[0] * x + h[1] * y + h[2]) * scale);
                long qy = Math.round((h[3] * x + h[4] * y + h[5]) * scale);
                dst[y * width + x] = bicubic(src, srcWidth, srcHeight,
                        (int) Math.floorDiv(qx, SUBPIXEL_STEPS), Math.floorMod(qx, SUBPIXEL_STEPS) / (double) SUBPIXEL_STEPS,
                        (int) Math.floorDiv(qy, SUBPIXEL_STEPS), Math.floorMod(qy, SUBPIXEL_STEPS) / (double) SUBPIXEL_STEPS);
            }
        }
        if (height >= 1.5 * width) {
            return rotateCounterClockwise(dst, width, height);
        }
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        out.setRGB(0, 0, width, height, dst, 0, width);
        return out;
    }

    /**
     * The 3×3 perspective transform mapping output pixel {@code (x, y)} to the source image: the output
     * rectangle's corners {@code (0,0), (w,0), (w,h), (0,h)} go to {@code corners}.
     */
    private static double[] homography(int width, int height, double[][] corners) {
        double[][] from = {{0, 0}, {width, 0}, {width, height}, {0, height}};
        double[][] a = new double[8][9];
        for (int i = 0; i < 4; i++) {
            double x = from[i][0], y = from[i][1], u = corners[i][0], v = corners[i][1];
            a[2 * i] = new double[]{x, y, 1, 0, 0, 0, -x * u, -y * u, u};
            a[2 * i + 1] = new double[]{0, 0, 0, x, y, 1, -x * v, -y * v, v};
        }
        double[] solution = solve(a);
        return new double[]{solution[0], solution[1], solution[2], solution[3], solution[4], solution[5],
                solution[6], solution[7], 1};
    }

    /** Gaussian elimination with partial pivoting on an augmented 8×9 system. */
    private static double[] solve(double[][] a) {
        int n = 8;
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) {
                    pivot = row;
                }
            }
            double[] swap = a[col];
            a[col] = a[pivot];
            a[pivot] = swap;
            if (Math.abs(a[col][col]) < 1e-12) {
                throw new IllegalArgumentException("Degenerate crop polygon");
            }
            for (int row = 0; row < n; row++) {
                if (row != col) {
                    double factor = a[row][col] / a[col][col];
                    for (int k = col; k <= n; k++) {
                        a[row][k] -= factor * a[col][k];
                    }
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = a[i][n] / a[i][i];
        }
        return x;
    }

    private static int bicubic(int[] src, int width, int height, int ix, double fx, int iy, double fy) {
        double[] wx = cubicWeights(fx);
        double[] wy = cubicWeights(fy);
        double r = 0, g = 0, b = 0;
        for (int j = 0; j < 4; j++) {
            int sy = Math.max(0, Math.min(iy - 1 + j, height - 1));
            double rowR = 0, rowG = 0, rowB = 0;
            for (int i = 0; i < 4; i++) {
                int sx = Math.max(0, Math.min(ix - 1 + i, width - 1));
                int rgb = src[sy * width + sx];
                rowR += wx[i] * ((rgb >> 16) & 0xFF);
                rowG += wx[i] * ((rgb >> 8) & 0xFF);
                rowB += wx[i] * (rgb & 0xFF);
            }
            r += wy[j] * rowR;
            g += wy[j] * rowG;
            b += wy[j] * rowB;
        }
        return (toByte(r) << 16) | (toByte(g) << 8) | toByte(b);
    }

    private static double[] cubicWeights(double t) {
        double w0 = ((CUBIC_A * (t + 1) - 5 * CUBIC_A) * (t + 1) + 8 * CUBIC_A) * (t + 1) - 4 * CUBIC_A;
        double w1 = ((CUBIC_A + 2) * t - (CUBIC_A + 3)) * t * t + 1;
        double w2 = ((CUBIC_A + 2) * (1 - t) - (CUBIC_A + 3)) * (1 - t) * (1 - t) + 1;
        return new double[]{w0, w1, w2, 1 - w0 - w1 - w2};
    }

    private static int toByte(double value) {
        return (int) Math.max(0, Math.min(255, Math.round(value)));
    }

    private static BufferedImage rotateCounterClockwise(int[] pixels, int width, int height) {
        BufferedImage out = new BufferedImage(height, width, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                out.setRGB(y, width - 1 - x, pixels[y * width + x]);
            }
        }
        return out;
    }

    /** NumPy's {@code linalg.norm} of the difference of two single-precision points. */
    private static float distance(double[] a, double[] b) {
        float dx = (float) a[0] - (float) b[0];
        float dy = (float) a[1] - (float) b[1];
        return (float) Math.sqrt(dx * dx + dy * dy);
    }
}
