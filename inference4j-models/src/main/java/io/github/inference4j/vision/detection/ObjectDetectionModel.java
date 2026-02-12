package io.github.inference4j.vision.detection;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

/**
 * Detects objects in images and returns labeled bounding boxes.
 */
public interface ObjectDetectionModel extends AutoCloseable {

    List<Detection> detect(BufferedImage image);

    List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold);

    List<Detection> detect(Path imagePath);

    List<Detection> detect(Path imagePath, float confidenceThreshold, float iouThreshold);

    @Override
    void close();
}
