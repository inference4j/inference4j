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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Plane geometry for text quadrilaterals, following OpenCV's {@code minAreaRect} and PaddleOCR's corner order.
 */
final class Quads {

    private Quads() {
    }

    /**
     * A rotated rectangle: its four corners (in no particular order) and its side lengths.
     */
    record RotatedRect(double[] xs, double[] ys, double width, double height) {

        double shortSide() {
            return Math.min(width, height);
        }
    }

    /**
     * Expands a polygon by {@code distance} on every side and returns the smallest rectangle around the result,
     * emulating PaddleOCR's {@code pyclipper} offset: integer input (truncated), round joins with Clipper's default
     * arc tolerance, integer output (rounded half away from zero). Rounding in integers this way, rather than
     * expanding exact coordinates, keeps box edges where PaddleOCR puts them, to the pixel.
     */
    static RotatedRect unclip(double[][] polygon, double distance) {
        int n = polygon.length;
        double[] px = new double[n];
        double[] py = new double[n];
        double cx = 0, cy = 0;
        for (int i = 0; i < n; i++) {
            px[i] = (long) polygon[i][0];
            py[i] = (long) polygon[i][1];
            cx += px[i] / n;
            cy += py[i] / n;
        }
        double arcTolerance = Math.min(0.25, Math.abs(distance) * 0.25);
        double stepsPerRadian = distance > 0
                ? Math.PI / Math.acos(1 - arcTolerance / Math.abs(distance)) / (2 * Math.PI)
                : 0;

        // Outward unit normal of each edge i -> i+1
        double[][] normals = new double[n][];
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double length = Math.hypot(dx, dy);
            double nx = length == 0 ? 0 : dy / length;
            double ny = length == 0 ? 0 : -dx / length;
            double mx = (px[i] + px[j]) / 2 - cx;
            double my = (py[i] + py[j]) / 2 - cy;
            if (nx * mx + ny * my < 0) {
                nx = -nx;
                ny = -ny;
            }
            normals[i] = new double[]{nx, ny};
        }

        List<double[]> points = new ArrayList<>();
        for (int k = 0; k < n; k++) {
            double[] before = normals[(k - 1 + n) % n];
            double[] after = normals[k];
            double angle = Math.atan2(before[0] * after[1] - before[1] * after[0],
                    before[0] * after[0] + before[1] * after[1]);
            int steps = Math.max((int) clipperRound(stepsPerRadian * Math.abs(angle)), 1);
            double step = angle / steps;
            for (int s = 0; s <= steps; s++) {
                double theta = Math.atan2(before[1], before[0]) + step * s;
                points.add(new double[]{clipperRound(px[k] + Math.cos(theta) * distance),
                        clipperRound(py[k] + Math.sin(theta) * distance)});
            }
        }
        double[] xs = new double[points.size()];
        double[] ys = new double[points.size()];
        for (int i = 0; i < points.size(); i++) {
            xs[i] = points.get(i)[0];
            ys[i] = points.get(i)[1];
        }
        return minAreaRectOfPolygon(xs, ys);
    }

    /** Clipper's {@code Round}: half away from zero. */
    private static long clipperRound(double value) {
        return value < 0 ? (long) (value - 0.5) : (long) (value + 0.5);
    }

    /**
     * OpenCV's {@code minAreaRect} for the outline of a connected area, as {@code findContours} would give it:
     * the hull starts at the vertex after the area's top-left pixel, as OpenCV orders it.
     */
    static RotatedRect minAreaRectOfArea(double[] xs, double[] ys) {
        double[][] hull = convexHull(xs, ys);
        int topLeft = 0;
        for (int i = 1; i < hull.length; i++) {
            if (hull[i][1] < hull[topLeft][1] || (hull[i][1] == hull[topLeft][1] && hull[i][0] < hull[topLeft][0])) {
                topLeft = i;
            }
        }
        return openCvMinAreaRect(rotate(hull, (topLeft + 1) % hull.length));
    }

    /**
     * OpenCV's {@code minAreaRect} for a polygon given point by point: OpenCV starts the hull at the polygon's first
     * point that lies on it (or its last, when the polygon runs the other way round).
     */
    static RotatedRect minAreaRectOfPolygon(double[] xs, double[] ys) {
        int n = xs.length;
        double[][] withIndex = new double[n][];
        for (int i = 0; i < n; i++) {
            withIndex[i] = new double[]{xs[i], ys[i], i};
        }
        double[][] hull = convexHullWithIndex(withIndex);
        int h = hull.length;
        if (h < 3) {
            return openCvMinAreaRect(hull);
        }
        int ascending = 0;
        int minAt = 0;
        int maxAt = 0;
        for (int i = 0; i < h; i++) {
            if (hull[(i + 1) % h][2] > hull[i][2]) {
                ascending++;
            }
            if (hull[i][2] < hull[minAt][2]) {
                minAt = i;
            }
            if (hull[i][2] > hull[maxAt][2]) {
                maxAt = i;
            }
        }
        int start = ascending >= h - 1 ? minAt : ascending <= 1 ? maxAt : 0;
        return openCvMinAreaRect(rotate(hull, start));
    }

    private static double[][] rotate(double[][] points, int start) {
        double[][] rotated = new double[points.length][];
        for (int i = 0; i < points.length; i++) {
            rotated[i] = points[(start + i) % points.length];
        }
        return rotated;
    }

    /**
     * A port of OpenCV 4.x's {@code minAreaRect} (rotating calipers, then {@code RotatedRect::points}) in single
     * precision, so corners carry the same rounding as OpenCV's: e.g. 54.99999 rather than 55, which PaddleOCR's
     * later integer truncation turns into a different pixel.
     */
    private static RotatedRect openCvMinAreaRect(double[][] hull) {
        int n = hull.length;
        float centerX, centerY, sizeW, sizeH, angle;
        if (n > 2) {
            float[] out = rotatingCalipers(hull);
            centerX = out[0] + (out[2] + out[4]) * 0.5f;
            centerY = out[1] + (out[3] + out[5]) * 0.5f;
            sizeW = (float) Math.sqrt((double) out[2] * out[2] + (double) out[3] * out[3]);
            sizeH = (float) Math.sqrt((double) out[4] * out[4] + (double) out[5] * out[5]);
            angle = (float) Math.atan2(out[3], out[2]);
        } else if (n == 2) {
            float x0 = (float) hull[0][0], y0 = (float) hull[0][1], x1 = (float) hull[1][0], y1 = (float) hull[1][1];
            centerX = (x0 + x1) * 0.5f;
            centerY = (y0 + y1) * 0.5f;
            double dx = x1 - x0;
            double dy = y1 - y0;
            sizeW = (float) Math.sqrt(dx * dx + dy * dy);
            sizeH = 0;
            angle = (float) Math.atan2(dy, dx);
        } else {
            centerX = (float) hull[0][0];
            centerY = (float) hull[0][1];
            sizeW = 0;
            sizeH = 0;
            angle = 0;
        }
        angle = (float) (angle * 180 / Math.PI);

        // RotatedRect::points
        double radians = angle * Math.PI / 180.;
        float b = (float) Math.cos(radians) * 0.5f;
        float a = (float) Math.sin(radians) * 0.5f;
        float[] px = new float[4];
        float[] py = new float[4];
        px[0] = centerX - a * sizeH - b * sizeW;
        py[0] = centerY + b * sizeH - a * sizeW;
        px[1] = centerX + a * sizeH - b * sizeW;
        py[1] = centerY - b * sizeH - a * sizeW;
        px[2] = 2 * centerX - px[0];
        py[2] = 2 * centerY - py[0];
        px[3] = 2 * centerX - px[1];
        py[3] = 2 * centerY - py[1];
        double[] xs = new double[4];
        double[] ys = new double[4];
        for (int i = 0; i < 4; i++) {
            xs[i] = px[i];
            ys[i] = py[i];
        }
        return new RotatedRect(xs, ys, sizeW, sizeH);
    }

    /** OpenCV's {@code rotatingCalipers} in {@code CALIPERS_MINAREARECT} mode: corner and two side vectors. */
    private static float[] rotatingCalipers(double[][] hull) {
        int n = hull.length;
        float[] x = new float[n];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) hull[i][0];
            y[i] = (float) hull[i][1];
        }
        float[] vx = new float[n];
        float[] vy = new float[n];
        float[] invLength = new float[n];
        int left = 0, bottom = 0, right = 0, top = 0;
        float leftX = x[0], rightX = x[0], topY = y[0], bottomY = y[0];
        float p0x = x[0], p0y = y[0];
        for (int i = 0; i < n; i++) {
            if (p0x < leftX) {
                leftX = p0x;
                left = i;
            }
            if (p0x > rightX) {
                rightX = p0x;
                right = i;
            }
            if (p0y > topY) {
                topY = p0y;
                top = i;
            }
            if (p0y < bottomY) {
                bottomY = p0y;
                bottom = i;
            }
            int next = i + 1 < n ? i + 1 : 0;
            double dx = x[next] - p0x;
            double dy = y[next] - p0y;
            vx[i] = (float) dx;
            vy[i] = (float) dy;
            invLength[i] = (float) (1. / Math.sqrt(dx * dx + dy * dy));
            p0x = x[next];
            p0y = y[next];
        }

        float orientation = 0;
        double ax = vx[n - 1], ay = vy[n - 1];
        for (int i = 0; i < n; i++) {
            double bx = vx[i], by = vy[i];
            double convexity = ax * by - ay * bx;
            if (convexity != 0) {
                orientation = convexity > 0 ? 1f : -1f;
                break;
            }
            ax = bx;
            ay = by;
        }
        float baseA = orientation;
        float baseB = 0;

        int[] seq = {bottom, right, top, left};
        float minArea = Float.MAX_VALUE;
        int bestLeft = 0, bestBottom = 0;
        float bestA = 0, bestB = 0, bestWidth = 0, bestHeight = 0;
        float[] rx = new float[4];
        float[] ry = new float[4];
        for (int k = 0; k < n; k++) {
            rx[0] = vx[seq[0]];
            ry[0] = vy[seq[0]];
            rx[1] = vy[seq[1]];       // rotate 90° clockwise
            ry[1] = -vx[seq[1]];
            rx[2] = -vx[seq[2]];      // rotate 180°
            ry[2] = -vy[seq[2]];
            rx[3] = -vy[seq[3]];      // rotate 90° counter-clockwise
            ry[3] = vx[seq[3]];
            int main = 0;
            for (int i = 1; i < 4; i++) {
                // first vector is to the right (clockwise) of the second
                float tx = ry[i];
                float ty = -rx[i];
                if (tx * rx[main] + ty * ry[main] < 0) {
                    main = i;
                }
            }
            int pindex = seq[main];
            float leadX = vx[pindex] * invLength[pindex];
            float leadY = vy[pindex] * invLength[pindex];
            switch (main) {
                case 0 -> {
                    baseA = leadX;
                    baseB = leadY;
                }
                case 1 -> {
                    baseA = leadY;
                    baseB = -leadX;
                }
                case 2 -> {
                    baseA = -leadX;
                    baseB = -leadY;
                }
                default -> {
                    baseA = -leadY;
                    baseB = leadX;
                }
            }
            seq[main] = seq[main] + 1 == n ? 0 : seq[main] + 1;

            float dx = x[seq[1]] - x[seq[3]];
            float dy = y[seq[1]] - y[seq[3]];
            float width = dx * baseA + dy * baseB;
            dx = x[seq[2]] - x[seq[0]];
            dy = y[seq[2]] - y[seq[0]];
            float height = -dx * baseB + dy * baseA;
            float area = width * height;
            if (area <= minArea) {
                minArea = area;
                bestLeft = seq[3];
                bestA = baseA;
                bestWidth = width;
                bestB = baseB;
                bestHeight = height;
                bestBottom = seq[0];
            }
        }
        float a1 = bestA, b1 = bestB, a2 = -bestB, b2 = bestA;
        float c1 = a1 * x[bestLeft] + y[bestLeft] * b1;
        float c2 = a2 * x[bestBottom] + y[bestBottom] * b2;
        float idet = 1f / (a1 * b2 - a2 * b1);
        float px = (c1 * b2 - c2 * b1) * idet;
        float py = (a1 * c2 - a2 * c1) * idet;
        return new float[]{px, py, a1 * bestWidth, b1 * bestWidth, a2 * bestHeight, b2 * bestHeight};
    }

    /** OpenCV's {@code contourArea} (absolute) of a polygon. */
    static double contourArea(double[][] polygon) {
        double area = 0;
        double[] previous = polygon[polygon.length - 1];
        for (double[] p : polygon) {
            area += previous[0] * p[1] - p[0] * previous[1];
            previous = p;
        }
        return Math.abs(area * 0.5);
    }

    /** OpenCV's {@code arcLength} of a closed polygon in single precision, as for float points. */
    static double arcLength(double[][] polygon) {
        double perimeter = 0;
        double[] previous = polygon[polygon.length - 1];
        for (double[] p : polygon) {
            float dx = (float) p[0] - (float) previous[0];
            float dy = (float) p[1] - (float) previous[1];
            perimeter += Math.sqrt(dx * dx + dy * dy);
            previous = p;
        }
        return perimeter;
    }

    /**
     * Orders a rectangle's corners as PaddleOCR's {@code get_mini_boxes} does: top-left, top-right, bottom-right,
     * bottom-left, choosing within each side pair by x and then by y.
     */
    static double[][] orderCorners(double[] xs, double[] ys) {
        Integer[] order = {0, 1, 2, 3};
        Arrays.sort(order, Comparator.comparingDouble(i -> xs[i]));   // stable, like Python's sorted
        int topLeft, bottomLeft, topRight, bottomRight;
        if (ys[order[1]] > ys[order[0]]) {
            topLeft = order[0];
            bottomLeft = order[1];
        } else {
            topLeft = order[1];
            bottomLeft = order[0];
        }
        if (ys[order[3]] > ys[order[2]]) {
            topRight = order[2];
            bottomRight = order[3];
        } else {
            topRight = order[3];
            bottomRight = order[2];
        }
        int[] picked = {topLeft, topRight, bottomRight, bottomLeft};
        double[][] corners = new double[4][];
        for (int i = 0; i < 4; i++) {
            corners[i] = new double[]{xs[picked[i]], ys[picked[i]]};
        }
        return corners;
    }

    /**
     * Andrew's monotone chain; returns the hull counter-clockwise (with the y axis pointing up, as OpenCV's
     * {@code convexHull} with {@code clockwise = false}), without collinear points or repeating the first point.
     */
    static double[][] convexHull(double[] xs, double[] ys) {
        int n = xs.length;
        double[][] points = new double[n][];
        for (int i = 0; i < n; i++) {
            points[i] = new double[]{xs[i], ys[i]};
        }
        return convexHullWithIndex(points);
    }

    /** {@link #convexHull} over points that may carry extra columns (e.g. their input index), kept as they are. */
    static double[][] convexHullWithIndex(double[][] input) {
        int n = input.length;
        double[][] points = input.clone();
        Arrays.sort(points, (p, q) -> p[0] != q[0] ? Double.compare(p[0], q[0]) : Double.compare(p[1], q[1]));
        if (n < 3) {
            return dedupe(points);
        }
        List<double[]> hull = new ArrayList<>();
        for (double[] p : points) {
            while (hull.size() >= 2 && cross(hull.get(hull.size() - 2), hull.get(hull.size() - 1), p) <= 0) {
                hull.remove(hull.size() - 1);
            }
            hull.add(p);
        }
        int lowerSize = hull.size();
        for (int i = n - 2; i >= 0; i--) {
            double[] p = points[i];
            while (hull.size() > lowerSize && cross(hull.get(hull.size() - 2), hull.get(hull.size() - 1), p) <= 0) {
                hull.remove(hull.size() - 1);
            }
            hull.add(p);
        }
        hull.remove(hull.size() - 1);
        return hull.isEmpty() ? new double[][]{points[0]} : hull.toArray(new double[0][]);
    }

    private static double[][] dedupe(double[][] sorted) {
        List<double[]> unique = new ArrayList<>();
        for (double[] p : sorted) {
            if (unique.isEmpty() || unique.get(unique.size() - 1)[0] != p[0] || unique.get(unique.size() - 1)[1] != p[1]) {
                unique.add(p);
            }
        }
        return unique.toArray(new double[0][]);
    }

    private static double cross(double[] o, double[] a, double[] b) {
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    }

    private static final int XY_SHIFT = 16;
    private static final long XY_ONE = 1L << XY_SHIFT;

    /**
     * OpenCV's {@code fillPoly} of one polygon with integer corners on a {@code width × height} mask: the outline
     * drawn as 8-connected lines plus a 16.16 fixed-point scanline fill, clipped to the mask.
     */
    static boolean[] fillPoly(long[][] points, int width, int height) {
        boolean[] mask = new boolean[width * height];
        List<long[]> edges = new ArrayList<>();   // y0, y1, x, dx
        long[] p0 = points[points.length - 1];
        for (long[] p1 : points) {
            long x0 = p0[0], y0 = p0[1], x1 = p1[0], y1 = p1[1];
            drawLine(mask, width, height, x0, y0, x1, y1);
            long c0x = x0 << XY_SHIFT, c0y = y0, c1x = x1 << XY_SHIFT, c1y = y1;
            if (x0 < 0 || x0 >= width || x1 < 0 || x1 >= width || y0 < 0 || y0 >= height || y1 < 0 || y1 >= height) {
                long[] clipped = {x0, y0, x1, y1};
                clipLine(width, height, clipped);
                if (clipped[1] != clipped[3]) {
                    c0x = clipped[0] << XY_SHIFT;
                    c0y = clipped[1];
                    c1x = clipped[2] << XY_SHIFT;
                    c1y = clipped[3];
                }
            } else {
                c0x += XY_ONE >> 1;
                c1x += XY_ONE >> 1;
            }
            if (y0 != y1) {
                long dx = (c1x - c0x) / (c1y - c0y);
                if (y0 < y1) {
                    edges.add(new long[]{y0, y1, c0x + (y0 - c0y) * dx, dx});
                } else {
                    edges.add(new long[]{y1, y0, c1x + (y1 - c1y) * dx, dx});
                }
            }
            p0 = p1;
        }
        if (edges.size() < 2) {
            return mask;
        }
        long yMin = Long.MAX_VALUE, yMax = Long.MIN_VALUE;
        for (long[] e : edges) {
            yMin = Math.min(yMin, e[0]);
            yMax = Math.max(yMax, e[1]);
        }
        yMax = Math.min(yMax, height);
        for (long y = yMin; y < yMax; y++) {
            List<Long> xs = new ArrayList<>();
            for (long[] e : edges) {
                if (e[0] <= y && y < e[1]) {
                    xs.add(e[2] + (y - e[0]) * e[3]);
                }
            }
            if (y < 0) {
                continue;
            }
            xs.sort(null);
            for (int i = 0; i + 1 < xs.size(); i += 2) {
                long x1 = xs.get(i) >> XY_SHIFT;
                long x2 = xs.get(i + 1) >> XY_SHIFT;
                if (x1 < width && x2 >= 0) {
                    for (long x = Math.max(0, x1); x <= Math.min(width - 1, x2); x++) {
                        mask[(int) (y * width + x)] = true;
                    }
                }
            }
        }
        return mask;
    }

    /** OpenCV's 8-connected {@code Line} (a {@code LineIterator} from left to right), clipped to the mask. */
    private static void drawLine(boolean[] mask, int width, int height, long ax, long ay, long bx, long by) {
        long[] pts = {ax, ay, bx, by};
        if (ax < 0 || ax >= width || bx < 0 || bx >= width || ay < 0 || ay >= height || by < 0 || by >= height) {
            if (!clipLine(width, height, pts)) {
                return;
            }
        }
        long x1 = pts[0], y1 = pts[1], x2 = pts[2], y2 = pts[3];
        long deltaX = 1, deltaY = 1;
        long dx = x2 - x1;
        long dy = y2 - y1;
        if (dx < 0) {   // left to right
            dx = -dx;
            dy = -dy;
            x1 = x2;
            y1 = y2;
        }
        if (dy < 0) {
            dy = -dy;
            deltaY = -1;
        }
        boolean vertical = dy > dx;
        if (vertical) {
            long t = dx;
            dx = dy;
            dy = t;
            t = deltaX;
            deltaX = deltaY;
            deltaY = t;
        }
        long err = dx - (dy + dy);
        long plusDelta = dx + dx;
        long minusDelta = -(dy + dy);
        // 8-connected: always step along the major axis, also along the minor one when err < 0
        long minusShift = deltaX, plusShift = 0, minusStep = 0, plusStep = deltaY;
        if (vertical) {
            long t = plusStep;
            plusStep = plusShift;
            plusShift = t;
            t = minusStep;
            minusStep = minusShift;
            minusShift = t;
        }
        long x = x1, y = y1;
        for (long i = 0; i <= dx; i++) {
            if (x >= 0 && x < width && y >= 0 && y < height) {
                mask[(int) (y * width + x)] = true;
            }
            boolean minus = err < 0;
            err += minusDelta + (minus ? plusDelta : 0);
            x += minusShift + (minus ? plusShift : 0);
            y += minusStep + (minus ? plusStep : 0);
        }
    }

    /** OpenCV's {@code clipLine}: clips {x1, y1, x2, y2} in place; false if the line misses the area. */
    private static boolean clipLine(int width, int height, long[] p) {
        long right = width - 1, bottom = height - 1;
        int c1 = code(p[0], p[1], right, bottom);
        int c2 = code(p[2], p[3], right, bottom);
        if ((c1 & c2) == 0 && (c1 | c2) != 0) {
            if ((c1 & 12) != 0) {
                long a = c1 < 8 ? 0 : bottom;
                p[0] += (long) ((double) (a - p[1]) * (p[2] - p[0]) / (p[3] - p[1]));
                p[1] = a;
                c1 = (p[0] < 0 ? 1 : 0) + (p[0] > right ? 2 : 0);
            }
            if ((c2 & 12) != 0) {
                long a = c2 < 8 ? 0 : bottom;
                p[2] += (long) ((double) (a - p[3]) * (p[2] - p[0]) / (p[3] - p[1]));
                p[3] = a;
                c2 = (p[2] < 0 ? 1 : 0) + (p[2] > right ? 2 : 0);
            }
            if ((c1 & c2) == 0 && (c1 | c2) != 0) {
                if (c1 != 0) {
                    long a = c1 == 1 ? 0 : right;
                    p[1] += (long) ((double) (a - p[0]) * (p[3] - p[1]) / (p[2] - p[0]));
                    p[0] = a;
                    c1 = 0;
                }
                if (c2 != 0) {
                    long a = c2 == 1 ? 0 : right;
                    p[3] += (long) ((double) (a - p[2]) * (p[3] - p[1]) / (p[2] - p[0]));
                    p[2] = a;
                    c2 = 0;
                }
            }
        }
        return (c1 | c2) == 0;
    }

    private static int code(long x, long y, long right, long bottom) {
        return (x < 0 ? 1 : 0) + (x > right ? 2 : 0) + (y < 0 ? 4 : 0) + (y > bottom ? 8 : 0);
    }
}
