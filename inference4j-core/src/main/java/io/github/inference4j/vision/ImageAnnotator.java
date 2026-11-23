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

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Draws bounding boxes and pose skeletons on images.
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
 * <p>Poses from a {@link PoseDetector} are drawn as a skeleton with {@link #annotatePoses(BufferedImage, List)}.
 *
 * @see BoundingBox
 */
public final class ImageAnnotator {

    private static final Color DEFAULT_COLOR = Color.GREEN;
    private static final float DEFAULT_STROKE_WIDTH = 2f;
    private static final float DEFAULT_MIN_KEYPOINT_CONFIDENCE = 0.3f;

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
     * Draws the skeleton of each pose on a copy of the image: a line for each {@link CocoJoint#edges() COCO bone}
     * and a dot for each joint, skipping joints with a confidence below 0.3 and the bones that touch them.
     *
     * @param image the source image (not modified)
     * @param poses poses to draw, usually from a {@link PoseDetector}
     * @return a new image with the skeletons drawn
     */
    public static BufferedImage annotatePoses(BufferedImage image, List<Pose> poses) {
        return annotatePoses(image, poses, CocoJoint.edges(), DEFAULT_MIN_KEYPOINT_CONFIDENCE, DEFAULT_COLOR,
                DEFAULT_STROKE_WIDTH);
    }

    /**
     * Draws the skeleton of each pose on a copy of the image with custom styling.
     *
     * @param image                 the source image (not modified)
     * @param poses                 poses to draw
     * @param edges                 bones to draw as lines, such as {@link CocoJoint#edges()}; a bone whose joints
     *                              a pose does not have is skipped
     * @param minKeypointConfidence joints scoring below this are not drawn, nor are the bones that touch them
     * @param color                 line and dot color
     * @param strokeWidth           line thickness in pixels; dots are a little wider
     * @return a new image with the skeletons drawn
     */
    public static BufferedImage annotatePoses(BufferedImage image, List<Pose> poses, List<Joint.Edge> edges,
                                              float minKeypointConfidence, Color color, float strokeWidth) {
        int type = image.getType() == BufferedImage.TYPE_CUSTOM
                ? (image.getColorModel().hasAlpha() ? BufferedImage.TYPE_INT_ARGB : BufferedImage.TYPE_INT_RGB)
                : image.getType();
        BufferedImage copy = new BufferedImage(image.getWidth(), image.getHeight(), type);
        Graphics2D g = copy.createGraphics();
        g.drawImage(image, 0, 0, null);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(color);
        g.setStroke(new BasicStroke(strokeWidth, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        float radius = Math.max(2f, strokeWidth * 1.5f);

        for (Pose pose : poses) {
            for (Joint.Edge edge : edges) {
                if (!pose.has(edge.from()) || !pose.has(edge.to())) {
                    continue;
                }
                Keypoint from = pose.keypoint(edge.from());
                Keypoint to = pose.keypoint(edge.to());
                if (from.confidence() >= minKeypointConfidence && to.confidence() >= minKeypointConfidence) {
                    g.draw(new Line2D.Float(from.x(), from.y(), to.x(), to.y()));
                }
            }
            for (Keypoint keypoint : pose.keypoints()) {
                if (keypoint.confidence() >= minKeypointConfidence) {
                    g.fill(new Ellipse2D.Float(keypoint.x() - radius, keypoint.y() - radius, 2 * radius,
                            2 * radius));
                }
            }
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
