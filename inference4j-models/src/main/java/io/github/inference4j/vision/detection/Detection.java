package io.github.inference4j.vision.detection;

/**
 * A single object detection result.
 *
 * @param box        bounding box in original image coordinates
 * @param label      human-readable class label (e.g., "person", "car")
 * @param classIndex zero-based class index in the model's label set
 * @param confidence detection confidence score (0–1)
 */
public record Detection(BoundingBox box, String label, int classIndex, float confidence) {
}
