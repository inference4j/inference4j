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
import java.util.Comparator;
import java.util.List;

/**
 * Turns CRAFT's two score maps into text regions, as the reference implementation's {@code getDetBoxes_core}
 * ({@code clovaai/CRAFT-pytorch}) does.
 *
 * <p>A pixel belongs to text where its text (region) score exceeds {@code lowTextThreshold} or its link (affinity)
 * score exceeds {@code linkThreshold}. Each 4-connected area of at least {@code minArea} pixels is kept if its
 * <em>highest</em> text score reaches {@code textThreshold}. Its link-only pixels are removed, the rest is dilated
 * in proportion to the area's thickness, and the minimum-area rectangle around it becomes the region: a rotated
 * box, or an axis-aligned one when it is close to square.
 */
public final class CraftPostProcessor {

    private final float textThreshold;
    private final float linkThreshold;
    private final float lowTextThreshold;
    private final int minArea;

    /**
     * @param textThreshold    minimum highest text score for an area to count as text
     * @param linkThreshold    link score above which a pixel joins characters into words
     * @param lowTextThreshold text score above which a pixel belongs to a character
     * @param minArea          minimum area in score-map pixels
     */
    public CraftPostProcessor(float textThreshold, float linkThreshold, float lowTextThreshold, int minArea) {
        this.textThreshold = textThreshold;
        this.linkThreshold = linkThreshold;
        this.lowTextThreshold = lowTextThreshold;
        this.minArea = minArea;
    }

    /**
     * Extracts text regions from CRAFT's score maps.
     *
     * @param textMap     text (region) scores, row by row
     * @param linkMap     link (affinity) scores, row by row
     * @param mapWidth    score map width
     * @param mapHeight   score map height
     * @param scale       image pixels per score-map pixel
     * @param imageWidth  width of the original image, which regions are clipped to
     * @param imageHeight height of the original image
     * @return the regions with four-corner outlines, highest confidence (highest text score) first
     */
    public List<TextRegion> process(float[] textMap, float[] linkMap, int mapWidth, int mapHeight,
                                    double scale, int imageWidth, int imageHeight) {
        int size = mapWidth * mapHeight;
        if (textMap.length != size || linkMap.length != size) {
            throw new IllegalArgumentException("Score maps must have " + mapWidth + " × " + mapHeight + " values");
        }
        boolean[] text = new boolean[size];
        boolean[] link = new boolean[size];
        boolean[] combined = new boolean[size];
        for (int i = 0; i < size; i++) {
            text[i] = textMap[i] > lowTextThreshold;
            link[i] = linkMap[i] > linkThreshold;
            combined[i] = text[i] || link[i];
        }

        int[] labels = new int[size];
        int[] queue = new int[size];
        List<TextRegion> regions = new ArrayList<>();
        int label = 0;
        for (int start = 0; start < size; start++) {
            if (!combined[start] || labels[start] != 0) {
                continue;
            }
            label++;
            int count = 0;
            int head = 0;
            int tail = 0;
            queue[tail++] = start;
            labels[start] = label;
            int left = mapWidth, top = mapHeight, right = -1, bottom = -1;
            float maxText = Float.NEGATIVE_INFINITY;
            while (head < tail) {
                int index = queue[head++];
                count++;
                int x = index % mapWidth;
                int y = index / mapWidth;
                left = Math.min(left, x);
                right = Math.max(right, x);
                top = Math.min(top, y);
                bottom = Math.max(bottom, y);
                maxText = Math.max(maxText, textMap[index]);
                int[] neighbors = {y > 0 ? index - mapWidth : -1, y < mapHeight - 1 ? index + mapWidth : -1,
                        x > 0 ? index - 1 : -1, x < mapWidth - 1 ? index + 1 : -1};
                for (int n : neighbors) {
                    if (n >= 0 && combined[n] && labels[n] == 0) {
                        labels[n] = label;
                        queue[tail++] = n;
                    }
                }
            }
            if (count < minArea || maxText < textThreshold) {
                continue;
            }
            TextRegion region = region(labels, label, text, link, mapWidth, mapHeight, count,
                    left, top, right - left + 1, bottom - top + 1, maxText, scale, imageWidth, imageHeight);
            if (region != null) {
                regions.add(region);
            }
        }
        regions.sort(Comparator.comparingDouble(TextRegion::confidence).reversed());
        return regions;
    }

    private TextRegion region(int[] labels, int label, boolean[] text, boolean[] link, int mapWidth, int mapHeight,
                              int count, int x, int y, int w, int h, float maxText,
                              double scale, int imageWidth, int imageHeight) {
        int niter = (int) (Math.sqrt((double) count * Math.min(w, h) / ((double) w * h)) * 2);
        int sx = Math.max(0, x - niter);
        int sy = Math.max(0, y - niter);
        int ex = Math.min(mapWidth, x + w + niter + 1);
        int ey = Math.min(mapHeight, y + h + niter + 1);
        int roiWidth = ex - sx;
        int roiHeight = ey - sy;

        // The area without its link-only pixels, within the region of interest
        boolean[] segment = new boolean[roiWidth * roiHeight];
        for (int yy = y; yy < y + h; yy++) {
            for (int xx = x; xx < x + w; xx++) {
                int index = yy * mapWidth + xx;
                if (labels[index] == label && !(link[index] && !text[index])) {
                    segment[(yy - sy) * roiWidth + (xx - sx)] = true;
                }
            }
        }
        segment = dilate(segment, roiWidth, roiHeight, niter + 1);

        // Leftmost and rightmost pixel of each row, in row order: the same hull, in the same order, as all pixels
        List<double[]> points = new ArrayList<>();
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE, minY = Integer.MAX_VALUE, maxY = Integer.MIN_VALUE;
        for (int yy = 0; yy < roiHeight; yy++) {
            int first = -1;
            int last = -1;
            for (int xx = 0; xx < roiWidth; xx++) {
                if (segment[yy * roiWidth + xx]) {
                    if (first < 0) {
                        first = xx;
                    }
                    last = xx;
                }
            }
            if (first >= 0) {
                points.add(new double[]{first + sx, yy + sy});
                if (last != first) {
                    points.add(new double[]{last + sx, yy + sy});
                }
                minX = Math.min(minX, first + sx);
                maxX = Math.max(maxX, last + sx);
                minY = Math.min(minY, yy + sy);
                maxY = Math.max(maxY, yy + sy);
            }
        }
        if (points.isEmpty()) {
            return null;
        }
        double[] xs = new double[points.size()];
        double[] ys = new double[points.size()];
        for (int i = 0; i < points.size(); i++) {
            xs[i] = points.get(i)[0];
            ys[i] = points.get(i)[1];
        }
        Quads.RotatedRect rect = Quads.minAreaRectOfPolygon(xs, ys);
        float[][] box = new float[4][];
        for (int i = 0; i < 4; i++) {
            box[i] = new float[]{(float) rect.xs()[i], (float) rect.ys()[i]};
        }

        // A nearly square box becomes axis-aligned ("align diamond-shape")
        float boxW = norm(box[0], box[1]);
        float boxH = norm(box[1], box[2]);
        float ratio = Math.max(boxW, boxH) / (Math.min(boxW, boxH) + 1e-5f);
        if (Math.abs(1 - ratio) <= 0.1f) {
            box = new float[][]{{minX, minY}, {maxX, minY}, {maxX, maxY}, {minX, maxY}};
        }

        // Clockwise from the corner with the smallest x + y
        int startIndex = 0;
        for (int i = 1; i < 4; i++) {
            if (box[i][0] + box[i][1] < box[startIndex][0] + box[startIndex][1]) {
                startIndex = i;
            }
        }
        List<Point> polygon = new ArrayList<>(4);
        float bx1 = Float.MAX_VALUE, by1 = Float.MAX_VALUE, bx2 = -Float.MAX_VALUE, by2 = -Float.MAX_VALUE;
        for (int i = 0; i < 4; i++) {
            float[] corner = box[(startIndex + i) % 4];
            float px = clamp((float) (corner[0] * scale), imageWidth);
            float py = clamp((float) (corner[1] * scale), imageHeight);
            polygon.add(new Point(px, py));
            bx1 = Math.min(bx1, px);
            by1 = Math.min(by1, py);
            bx2 = Math.max(bx2, px);
            by2 = Math.max(by2, py);
        }
        return new TextRegion(new BoundingBox(bx1, by1, bx2, by2), polygon, maxText);
    }

    /**
     * OpenCV's {@code dilate} with a {@code k × k} rectangle anchored at its center ({@code k / 2}); pixels outside
     * the region of interest do not count.
     */
    private static boolean[] dilate(boolean[] src, int width, int height, int k) {
        if (k <= 1) {
            return src;
        }
        int anchor = k / 2;
        boolean[] rows = new boolean[src.length];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean any = false;
                for (int i = Math.max(0, x - anchor); i <= Math.min(width - 1, x - anchor + k - 1) && !any; i++) {
                    any = src[y * width + i];
                }
                rows[y * width + x] = any;
            }
        }
        boolean[] out = new boolean[src.length];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean any = false;
                for (int j = Math.max(0, y - anchor); j <= Math.min(height - 1, y - anchor + k - 1) && !any; j++) {
                    any = rows[j * width + x];
                }
                out[y * width + x] = any;
            }
        }
        return out;
    }

    /** NumPy's {@code linalg.norm} of the difference of two single-precision points. */
    private static float norm(float[] a, float[] b) {
        float dx = a[0] - b[0];
        float dy = a[1] - b[1];
        return (float) Math.sqrt(dx * dx + dy * dy);
    }

    private static float clamp(float value, int max) {
        return Math.max(0, Math.min(value, max));
    }
}
