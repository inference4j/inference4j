package io.github.inference4j.vision.detection;

/**
 * An axis-aligned bounding box in pixel coordinates.
 *
 * <p>Coordinates are in {@code (x1, y1, x2, y2)} format where
 * {@code (x1, y1)} is the top-left corner and {@code (x2, y2)} is
 * the bottom-right corner, both in original image pixels.
 *
 * @param x1 left edge (pixels)
 * @param y1 top edge (pixels)
 * @param x2 right edge (pixels)
 * @param y2 bottom edge (pixels)
 */
public record BoundingBox(float x1, float y1, float x2, float y2) {

    /** Width of the bounding box in pixels. */
    public float width() {
        return x2 - x1;
    }

    /** Height of the bounding box in pixels. */
    public float height() {
        return y2 - y1;
    }

    /** Area of the bounding box in square pixels. */
    public float area() {
        return width() * height();
    }
}
