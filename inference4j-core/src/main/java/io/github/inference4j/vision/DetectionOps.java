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

import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.processing.MathOps;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Building blocks shared by the object detectors: letterboxing, image-to-tensor conversion, non-maximum suppression
 * and the decoding of DETR-style query outputs. Custom detectors can use them to produce results that follow the
 * {@link ObjectDetector} contract.
 */
public final class DetectionOps {

    private DetectionOps() {
    }

    // --- Letterbox ---

    /**
     * How an image is scaled into a square model input while keeping its aspect ratio, and how boxes map back.
     *
     * @param scale        factor from original to model-input pixels
     * @param padX         horizontal padding before the image, in model-input pixels
     * @param padY         vertical padding before the image, in model-input pixels
     * @param scaledWidth  width of the scaled image inside the input
     * @param scaledHeight height of the scaled image inside the input
     * @param targetSize   side of the square model input
     */
    public record Letterbox(float scale, float padX, float padY, int scaledWidth, int scaledHeight, int targetSize) {

        /** The image centered in the input, padding split evenly on both sides (Ultralytics). */
        public static Letterbox centered(int width, int height, int targetSize) {
            float scale = Math.min((float) targetSize / width, (float) targetSize / height);
            int scaledWidth = Math.round(width * scale);
            int scaledHeight = Math.round(height * scale);
            return new Letterbox(scale, (targetSize - scaledWidth) / 2f, (targetSize - scaledHeight) / 2f,
                    scaledWidth, scaledHeight, targetSize);
        }

        /**
         * The image in the top-left corner of the input, padding on the right and bottom only, its scaled size
         * truncated rather than rounded (YOLOX's {@code preproc}).
         */
        public static Letterbox topLeft(int width, int height, int targetSize) {
            // In double, as Python computes it: in float, 352 * (416 / 352f) is 415.99997 and truncates to 415
            double scale = Math.min((double) targetSize / width, (double) targetSize / height);
            return new Letterbox((float) scale, 0f, 0f, (int) (width * scale), (int) (height * scale), targetSize);
        }

        /** Draws the scaled image on a square canvas filled with {@code padding}. */
        public BufferedImage apply(BufferedImage image, Color padding) {
            BufferedImage result = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = result.createGraphics();
            g.setColor(padding);
            g.fillRect(0, 0, targetSize, targetSize);
            g.drawImage(image, Math.round(padX), Math.round(padY), scaledWidth, scaledHeight, null);
            g.dispose();
            return result;
        }

        /**
         * Maps a box in model-input pixels back to the original image, clamped to its bounds.
         *
         * @param box    {@code [x1, y1, x2, y2]} in model-input pixels
         * @param width  original image width
         * @param height original image height
         */
        public BoundingBox toOriginal(float[] box, int width, int height) {
            return new BoundingBox(
                    clamp((box[0] - padX) / scale, width),
                    clamp((box[1] - padY) / scale, height),
                    clamp((box[2] - padX) / scale, width),
                    clamp((box[3] - padY) / scale, height));
        }
    }

    // --- Image to tensor ---

    /** The packed RGB pixels of an image, row by row. */
    public static int[] pixels(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        return image.getRGB(0, 0, width, height, null, 0, width);
    }

    /**
     * Converts packed RGB pixels to planar (CHW) floats in RGB order, each channel scaled to [0, 1].
     *
     * @return {@code 3 * width * height} floats: the red plane, then green, then blue
     */
    public static float[] toChw(int[] pixels, int width, int height) {
        int plane = width * height;
        float[] data = new float[3 * plane];
        for (int i = 0; i < plane; i++) {
            int rgb = pixels[i];
            data[i] = ((rgb >> 16) & 0xFF) / 255f;
            data[plane + i] = ((rgb >> 8) & 0xFF) / 255f;
            data[2 * plane + i] = (rgb & 0xFF) / 255f;
        }
        return data;
    }

    /**
     * Converts packed RGB pixels to planar (CHW) floats, each channel scaled to [0, 1], then normalized as
     * {@code (value - mean) / std}.
     *
     * @param mean per-channel mean, in the output channel order
     * @param std  per-channel standard deviation, in the output channel order
     * @param bgr  whether the output planes are blue, green, red instead of red, green, blue
     */
    public static float[] toChw(int[] pixels, int width, int height, float[] mean, float[] std, boolean bgr) {
        if (mean.length != 3 || std.length != 3) {
            throw new IllegalArgumentException("mean and std need three values, one per channel");
        }
        float[] data = toChw(pixels, width, height);
        int plane = width * height;
        if (bgr) {
            for (int i = 0; i < plane; i++) {
                float red = data[i];
                data[i] = data[2 * plane + i];
                data[2 * plane + i] = red;
            }
        }
        for (int c = 0; c < 3; c++) {
            for (int i = c * plane; i < (c + 1) * plane; i++) {
                data[i] = (data[i] - mean[c]) / std[c];
            }
        }
        return data;
    }

    // --- Non-maximum suppression ---

    /**
     * Keeps the most confident detections and drops those that overlap a kept one by more than
     * {@code iouThreshold}.
     *
     * @param classAware    whether only detections of the same class suppress each other
     * @param maxDetections the most detections returned
     * @return the kept detections, by descending confidence
     */
    public static List<Detection> nms(List<Detection> detections, float iouThreshold, boolean classAware,
                                      int maxDetections) {
        if (detections.isEmpty()) {
            return List.of();
        }
        int n = detections.size();
        float[] boxes = new float[n * 4];
        float[] scores = new float[n];
        // Class-aware NMS in one pass: shift each class to its own region, so boxes of different classes never overlap
        float offset = 0f;
        if (classAware) {
            for (Detection d : detections) {
                offset = Math.max(offset, Math.max(Math.abs(d.box().x2()), Math.abs(d.box().y2())));
            }
            offset += 1f;
        }
        for (int i = 0; i < n; i++) {
            Detection d = detections.get(i);
            float shift = classAware ? d.classIndex() * offset : 0f;
            boxes[i * 4] = d.box().x1() + shift;
            boxes[i * 4 + 1] = d.box().y1() + shift;
            boxes[i * 4 + 2] = d.box().x2() + shift;
            boxes[i * 4 + 3] = d.box().y2() + shift;
            scores[i] = d.confidence();
        }
        int[] kept = MathOps.nms(boxes, scores, iouThreshold);
        List<Detection> result = new ArrayList<>(Math.min(kept.length, maxDetections));
        for (int i = 0; i < Math.min(kept.length, maxDetections); i++) {
            result.add(detections.get(kept[i]));
        }
        return result;
    }

    // --- DETR-style query decoding ---

    /** How class scores of the object queries become detections. */
    public enum QueryDecoding {
        /**
         * The best (query, class) pairs over all queries, scores above the threshold; one query can yield several
         * classes. HuggingFace {@code post_process_object_detection} for RT-DETR and D-FINE.
         */
        TOP_PAIRS,
        /** Each query's best class only, scores at or above the threshold (YOLO26). */
        BEST_CLASS_PER_QUERY
    }

    /**
     * Decodes the outputs of a DETR-style detector: per-query class logits and normalized {@code cx, cy, w, h}
     * boxes. Scores are the sigmoid of the logits; boxes are scaled to the image and clamped to it.
     *
     * @param logits        {@code queries * classes} class logits
     * @param boxes         {@code queries * 4} normalized boxes
     * @param width         original image width
     * @param height        original image height
     * @param maxDetections the most detections returned; at most {@code queries} in any case
     * @return detections by descending confidence
     */
    public static List<Detection> decodeQueries(float[] logits, float[] boxes, int queries, int classes,
                                                float threshold, int width, int height, Labels labels,
                                                QueryDecoding decoding, int maxDetections) {
        List<float[]> candidates = new ArrayList<>();   // score, query, class
        for (int q = 0; q < queries; q++) {
            if (decoding == QueryDecoding.TOP_PAIRS) {
                for (int c = 0; c < classes; c++) {
                    float score = sigmoid(logits[q * classes + c]);
                    if (score > threshold) {
                        candidates.add(new float[]{score, q, c});
                    }
                }
            } else {
                int bestClass = -1;
                float bestScore = -1f;
                for (int c = 0; c < classes; c++) {
                    float score = sigmoid(logits[q * classes + c]);
                    if (score > bestScore) {
                        bestScore = score;
                        bestClass = c;
                    }
                }
                if (bestScore >= threshold) {
                    candidates.add(new float[]{bestScore, q, bestClass});
                }
            }
        }
        // Stable sort: equal scores keep query order
        candidates.sort((a, b) -> Float.compare(b[0], a[0]));
        int count = Math.min(candidates.size(), Math.min(queries, maxDetections));
        List<Detection> detections = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            float[] candidate = candidates.get(i);
            int q = (int) candidate[1];
            int c = (int) candidate[2];
            float halfW = boxes[q * 4 + 2] / 2f;
            float halfH = boxes[q * 4 + 3] / 2f;
            float cx = boxes[q * 4];
            float cy = boxes[q * 4 + 1];
            BoundingBox box = new BoundingBox(
                    clamp((cx - halfW) * width, width),
                    clamp((cy - halfH) * height, height),
                    clamp((cx + halfW) * width, width),
                    clamp((cy + halfH) * height, height));
            detections.add(new Detection(box, labels.get(c), c, candidate[0]));
        }
        return detections;
    }

    private static float sigmoid(float logit) {
        return (float) (1.0 / (1.0 + Math.exp(-logit)));
    }

    private static float clamp(float value, int max) {
        return Math.max(0, Math.min(value, max));
    }
}
