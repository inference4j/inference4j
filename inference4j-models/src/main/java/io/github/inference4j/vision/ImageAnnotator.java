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

import io.github.inference4j.vision.detection.BoundingBox;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Draws bounding box annotations on images.
 *
 * <p>Works with any detection result type that provides a {@link BoundingBox} —
 * extract boxes with {@code detections.stream().map(Detection::box).toList()} or
 * {@code regions.stream().map(TextRegion::box).toList()}.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * List<BoundingBox> boxes = regions.stream().map(TextRegion::box).toList();
 * BufferedImage annotated = ImageAnnotator.annotate(image, boxes);
 * ImageAnnotator.save(annotated, Path.of("output.jpg"));
 * }</pre>
 *
 * @see BoundingBox
 */
public final class ImageAnnotator {

    private static final Color DEFAULT_COLOR = Color.GREEN;
    private static final float DEFAULT_STROKE_WIDTH = 2f;

    private ImageAnnotator() {
    }

    /**
     * Draws bounding box rectangles on a copy of the image.
     *
     * @param image the source image (not modified)
     * @param boxes bounding boxes to draw
     * @return a new image with rectangles drawn
     */
    public static BufferedImage annotate(BufferedImage image, List<BoundingBox> boxes) {
        return annotate(image, boxes, DEFAULT_COLOR, DEFAULT_STROKE_WIDTH);
    }

    /**
     * Draws bounding box rectangles on a copy of the image with custom styling.
     *
     * @param image       the source image (not modified)
     * @param boxes       bounding boxes to draw
     * @param color       rectangle color
     * @param strokeWidth line thickness in pixels
     * @return a new image with rectangles drawn
     */
    public static BufferedImage annotate(BufferedImage image, List<BoundingBox> boxes,
                                         Color color, float strokeWidth) {
        BufferedImage copy = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
        Graphics2D g = copy.createGraphics();
        g.drawImage(image, 0, 0, null);

        g.setColor(color);
        g.setStroke(new BasicStroke(strokeWidth));

        for (BoundingBox box : boxes) {
            int x = Math.round(box.x1());
            int y = Math.round(box.y1());
            int w = Math.round(box.width());
            int h = Math.round(box.height());
            g.drawRect(x, y, w, h);
        }

        g.dispose();
        return copy;
    }

    /**
     * Saves an image to a file. The format is inferred from the file extension.
     *
     * @param image      the image to save
     * @param outputPath destination path ({@code .jpg} or {@code .png})
     * @throws IOException if writing fails
     */
    public static void save(BufferedImage image, Path outputPath) throws IOException {
        String filename = outputPath.getFileName().toString();
        String format = filename.endsWith(".png") ? "png" : "jpg";
        ImageIO.write(image, format, outputPath.toFile());
    }
}
