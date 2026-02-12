package io.github.inference4j.vision.classification;

/**
 * A single classification result with label, class index, and confidence score.
 */
public record Classification(String label, int index, float confidence) {
}
