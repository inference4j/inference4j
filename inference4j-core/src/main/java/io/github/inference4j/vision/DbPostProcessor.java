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
import java.util.List;

/**
 * Turns the probability map of a DB (Differentiable Binarization) text detector, such as PaddleOCR's, into
 * rotated text regions.
 *
 * <p>Follows PaddleOCR's {@code DBPostProcess} with quadrilateral boxes: the map is binarized at
 * {@code threshold}; each connected area becomes its minimum-area rectangle, scored by the mean probability
 * inside it and dropped below {@code boxThreshold}; surviving rectangles are expanded ("unclipped") by
 * {@code area × unclipRatio / perimeter} on every side, since DB maps cover a shrunken core of each line, and
 * scaled to the original image.
 */
public final class DbPostProcessor {

    private static final int MIN_SIZE = 3;

    private final float threshold;
    private final float boxThreshold;
    private final float unclipRatio;
    private final int maxCandidates;

    /**
     * @param threshold     probability above which a map pixel counts as text
     * @param boxThreshold  minimum mean probability inside a region for it to be kept
     * @param unclipRatio   how far regions are expanded, relative to their area over perimeter
     * @param maxCandidates maximum number of connected areas considered
     */
    public DbPostProcessor(float threshold, float boxThreshold, float unclipRatio, int maxCandidates) {
        this.threshold = threshold;
        this.boxThreshold = boxThreshold;
        this.unclipRatio = unclipRatio;
        this.maxCandidates = maxCandidates;
    }

    /**
     * Extracts text regions from a probability map.
     *
     * @param probabilities the map, row by row, {@code mapWidth × mapHeight} values in [0, 1]
     * @param mapWidth      map width
     * @param mapHeight     map height
     * @param imageWidth    width of the original image, which the regions are scaled to
     * @param imageHeight   height of the original image
     * @return the regions, with four-corner polygons clockwise from the top-left, in no particular order
     */
    public List<TextRegion> process(float[] probabilities, int mapWidth, int mapHeight,
                                    int imageWidth, int imageHeight) {
        if (probabilities.length != mapWidth * mapHeight) {
            throw new IllegalArgumentException("Map has " + probabilities.length + " values, expected "
                    + mapWidth + " × " + mapHeight);
        }
        double scaleX = (double) imageWidth / mapWidth;
        double scaleY = (double) imageHeight / mapHeight;

        List<TextRegion> regions = new ArrayList<>();
        int candidates = 0;
        int[] labels = new int[mapWidth * mapHeight];
        int[] queue = new int[mapWidth * mapHeight];
        int label = 0;
        for (int start = 0; start < labels.length && candidates < maxCandidates; start++) {
            if (labels[start] != 0 || probabilities[start] <= threshold) {
                continue;
            }
            label++;
            candidates++;
            double[][] outline = component(probabilities, mapWidth, mapHeight, start, label, labels, queue);
            TextRegion region = region(probabilities, mapWidth, mapHeight, outline, scaleX, scaleY,
                    imageWidth, imageHeight);
            if (region != null) {
                regions.add(region);
            }
        }
        return regions;
    }

    /**
     * Labels one 8-connected area and returns its leftmost and rightmost pixel in every row: enough for its
     * convex hull.
     */
    private double[][] component(float[] probabilities, int width, int height, int start, int label,
                                 int[] labels, int[] queue) {
        int minY = start / width;
        int maxY = minY;
        int[] rowMin = new int[height];
        int[] rowMax = new int[height];
        java.util.Arrays.fill(rowMin, Integer.MAX_VALUE);
        java.util.Arrays.fill(rowMax, Integer.MIN_VALUE);

        int head = 0;
        int tail = 0;
        queue[tail++] = start;
        labels[start] = label;
        while (head < tail) {
            int index = queue[head++];
            int x = index % width;
            int y = index / width;
            rowMin[y] = Math.min(rowMin[y], x);
            rowMax[y] = Math.max(rowMax[y], x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
            for (int dy = -1; dy <= 1; dy++) {
                int ny = y + dy;
                if (ny < 0 || ny >= height) {
                    continue;
                }
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    if ((dx == 0 && dy == 0) || nx < 0 || nx >= width) {
                        continue;
                    }
                    int neighbor = ny * width + nx;
                    if (labels[neighbor] == 0 && probabilities[neighbor] > threshold) {
                        labels[neighbor] = label;
                        queue[tail++] = neighbor;
                    }
                }
            }
        }
        List<double[]> points = new ArrayList<>();
        for (int y = minY; y <= maxY; y++) {
            if (rowMin[y] != Integer.MAX_VALUE) {
                points.add(new double[]{rowMin[y], y});
                points.add(new double[]{rowMax[y], y});
            }
        }
        double[] xs = new double[points.size()];
        double[] ys = new double[points.size()];
        for (int i = 0; i < points.size(); i++) {
            xs[i] = points.get(i)[0];
            ys[i] = points.get(i)[1];
        }
        return new double[][]{xs, ys};
    }

    private TextRegion region(float[] probabilities, int mapWidth, int mapHeight, double[][] outline,
                              double scaleX, double scaleY, int imageWidth, int imageHeight) {
        Quads.RotatedRect rect = Quads.minAreaRectOfArea(outline[0], outline[1]);
        if (rect.shortSide() < MIN_SIZE) {
            return null;
        }
        double[][] box = Quads.orderCorners(rect.xs(), rect.ys());
        float score = meanInside(probabilities, mapWidth, mapHeight, box);
        if (score < boxThreshold) {
            return null;
        }

        double area = Quads.contourArea(box);
        double perimeter = Quads.arcLength(box);
        Quads.RotatedRect expanded = Quads.unclip(box, area * unclipRatio / perimeter);
        if (expanded.shortSide() < MIN_SIZE + 2) {
            return null;
        }
        double[][] corners = Quads.orderCorners(expanded.xs(), expanded.ys());

        List<Point> polygon = new ArrayList<>(4);
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE, maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (double[] corner : corners) {
            // Math.rint rounds halves to even, like Python's round()
            float x = (float) Math.max(0, Math.min(Math.rint(corner[0] * scaleX), imageWidth));
            float y = (float) Math.max(0, Math.min(Math.rint(corner[1] * scaleY), imageHeight));
            polygon.add(new Point(x, y));
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }
        return new TextRegion(new BoundingBox(minX, minY, maxX, maxY), polygon, score);
    }

    /** Mean probability of the map pixels inside the box, as PaddleOCR's {@code box_score_fast}. */
    private static float meanInside(float[] probabilities, int width, int height, double[][] box) {
        double minBx = Double.MAX_VALUE, maxBx = -Double.MAX_VALUE, minBy = Double.MAX_VALUE, maxBy = -Double.MAX_VALUE;
        for (double[] p : box) {
            minBx = Math.min(minBx, p[0]);
            maxBx = Math.max(maxBx, p[0]);
            minBy = Math.min(minBy, p[1]);
            maxBy = Math.max(maxBy, p[1]);
        }
        int xMin = clamp((int) Math.floor(minBx), width - 1);
        int xMax = clamp((int) Math.ceil(maxBx), width - 1);
        int yMin = clamp((int) Math.floor(minBy), height - 1);
        int yMax = clamp((int) Math.ceil(maxBy), height - 1);

        // OpenCV fills the polygon with its corners, relative to the clipped box, truncated to integers
        int maskWidth = xMax - xMin + 1;
        int maskHeight = yMax - yMin + 1;
        long[][] polygon = new long[4][];
        for (int i = 0; i < 4; i++) {
            polygon[i] = new long[]{(long) (box[i][0] - xMin), (long) (box[i][1] - yMin)};
        }
        boolean[] mask = Quads.fillPoly(polygon, maskWidth, maskHeight);
        double sum = 0;
        int count = 0;
        for (int y = 0; y < maskHeight; y++) {
            for (int x = 0; x < maskWidth; x++) {
                if (mask[y * maskWidth + x]) {
                    sum += probabilities[(yMin + y) * width + xMin + x];
                    count++;
                }
            }
        }
        return count == 0 ? 0f : (float) (sum / count);
    }

    private static int clamp(int value, int max) {
        return Math.max(0, Math.min(value, max));
    }
}
