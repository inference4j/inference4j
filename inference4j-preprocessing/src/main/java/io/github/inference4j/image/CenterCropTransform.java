package io.github.inference4j.image;

import java.awt.image.BufferedImage;

/**
 * Crops the center region of an image to the specified dimensions.
 */
public class CenterCropTransform implements ImageTransform {

    private final int width;
    private final int height;

    public CenterCropTransform(int width, int height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public BufferedImage apply(BufferedImage image) {
        int x = (image.getWidth() - width) / 2;
        int y = (image.getHeight() - height) / 2;
        return image.getSubimage(x, y, width, height);
    }
}
