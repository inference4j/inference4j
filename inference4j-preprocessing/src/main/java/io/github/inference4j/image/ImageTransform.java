package io.github.inference4j.image;

import java.awt.image.BufferedImage;

/**
 * A single image transformation step.
 */
@FunctionalInterface
public interface ImageTransform {

    BufferedImage apply(BufferedImage image);
}
