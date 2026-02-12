package io.github.inference4j.image;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Maps class indices to human-readable labels.
 *
 * <p>Provides presets for common datasets:
 * <pre>{@code
 * Labels labels = Labels.imagenet();  // 1000 ImageNet classes
 * Labels labels = Labels.coco();      // 80 COCO classes
 * }</pre>
 *
 * <p>Or load from file / custom list:
 * <pre>{@code
 * Labels labels = Labels.fromFile(Path.of("my-labels.txt"));
 * Labels labels = Labels.of(List.of("cat", "dog", "bird"));
 * }</pre>
 */
public class Labels {

    private final List<String> labels;

    private Labels(List<String> labels) {
        this.labels = List.copyOf(labels);
    }

    /**
     * Returns the 1000 ImageNet class labels.
     */
    public static Labels imagenet() {
        return fromResource("/imagenet-labels.txt");
    }

    /**
     * Returns the 80 COCO class labels.
     */
    public static Labels coco() {
        return fromResource("/coco-labels.txt");
    }

    /**
     * Loads labels from a text file, one label per line.
     */
    public static Labels fromFile(Path path) {
        try {
            List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
            return new Labels(lines);
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to read labels from " + path, e);
        }
    }

    /**
     * Creates labels from the given list.
     */
    public static Labels of(List<String> labels) {
        return new Labels(labels);
    }

    /**
     * Returns the label at the given index.
     *
     * @throws IndexOutOfBoundsException if the index is out of range
     */
    public String get(int index) {
        return labels.get(index);
    }

    /**
     * Returns the total number of labels.
     */
    public int size() {
        return labels.size();
    }

    private static Labels fromResource(String resourcePath) {
        try (InputStream is = Labels.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new IllegalStateException("Resource not found: " + resourcePath);
            }
            List<String> lines = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    lines.add(line);
                }
            }
            return new Labels(lines);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load labels from resource " + resourcePath, e);
        }
    }
}
