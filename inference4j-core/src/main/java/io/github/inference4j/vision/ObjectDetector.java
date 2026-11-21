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

import io.github.inference4j.Detector;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

/**
 * Detects objects in images and returns labeled bounding boxes.
 *
 * <h2>Swap contract</h2>
 * <p>Every detector in inference4j follows the same contract, so one can replace another without code changes:
 * <ul>
 *   <li>boxes are in pixel coordinates of the original image, with {@code x1 <= x2} and {@code y1 <= y2}, inside
 *       the image;</li>
 *   <li>detections are sorted by descending confidence;</li>
 *   <li>with the default labels, {@code classIndex} is a contiguous COCO index (0 to 79) and {@code label} is
 *       {@code Labels.coco().get(classIndex)}; models trained with other COCO ids are remapped by their preset;</li>
 *   <li>{@code Path} and {@link BufferedImage} inputs give the same detections for the same image.</li>
 * </ul>
 *
 * <p>The IoU threshold applies only to detectors that run non-maximum suppression ({@link YoloV8Detector}).
 * NMS-free detectors ({@link Yolo26Detector}, {@link DetrDetector}) ignore it.
 */
public interface ObjectDetector extends Detector<BufferedImage, Detection> {

    /** Detects objects with the configured thresholds. */
    @Override
    List<Detection> detect(BufferedImage image);

    /**
     * Detects objects with a confidence threshold other than the configured one.
     *
     * <p>The detectors in inference4j keep their configured IoU threshold. This default implementation, for custom
     * detectors, calls {@link #detect(BufferedImage, float, float)} with an IoU threshold of 0.45.
     */
    default List<Detection> detect(BufferedImage image, float confidenceThreshold) {
        return detect(image, confidenceThreshold, 0.45f);
    }

    /**
     * Detects objects with the given thresholds. Detectors without non-maximum suppression ignore
     * {@code iouThreshold}.
     */
    List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold);

    /** Reads the image file and detects objects with the configured thresholds. */
    default List<Detection> detect(Path imagePath) {
        return detect(ImageFiles.read(imagePath));
    }

    /** Reads the image file and detects objects with a confidence threshold other than the configured one. */
    default List<Detection> detect(Path imagePath, float confidenceThreshold) {
        return detect(ImageFiles.read(imagePath), confidenceThreshold);
    }

    /** Reads the image file and detects objects with the given thresholds. */
    default List<Detection> detect(Path imagePath, float confidenceThreshold, float iouThreshold) {
        return detect(ImageFiles.read(imagePath), confidenceThreshold, iouThreshold);
    }

    @Override
    void close();
}
