package io.github.inference4j.vision.classification;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

/**
 * Classifies images into labeled categories.
 */
public interface ImageClassificationModel extends AutoCloseable {

    List<Classification> classify(BufferedImage image);

    List<Classification> classify(BufferedImage image, int topK);

    List<Classification> classify(Path imagePath);

    List<Classification> classify(Path imagePath, int topK);

    @Override
    void close();
}
