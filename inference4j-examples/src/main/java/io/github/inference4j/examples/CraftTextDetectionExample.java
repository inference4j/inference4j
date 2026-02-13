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

package io.github.inference4j.examples;

import io.github.inference4j.vision.ImageAnnotator;
import io.github.inference4j.vision.detection.BoundingBox;
import io.github.inference4j.vision.detection.Craft;
import io.github.inference4j.vision.detection.TextRegion;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Demonstrates text detection with CRAFT.
 *
 * Requires an ONNX model and a sample image — see inference4j-examples/README.md for setup.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.CraftTextDetectionExample
 */
public class CraftTextDetectionExample {

    public static void main(String[] args) throws IOException {
        Path imagePath = Path.of("assets/images/product_placement.jpg");

        System.out.println("=== CRAFT Text Detection ===");
        try (Craft craft = Craft.fromPretrained("assets/models/craft")) {
            System.out.println("CRAFT model loaded successfully.");
            System.out.println();

            // The default thresholds (textThreshold=0.7, lowTextThreshold=0.4) match the
            // original CRAFT paper and work well for clean document scans. For real-world
            // images (product labels, street signs, etc.), lower thresholds are needed.
            List<TextRegion> regions = craft.detect(imagePath, 0.4f, 0.3f);

            System.out.println("Detected text regions (textThreshold=0.4, lowTextThreshold=0.3):");
            printRegions(regions);

            // --- Save annotated image using ImageAnnotator ---
            Path annotatedPath = annotatedPath(imagePath);
            BufferedImage image = ImageIO.read(imagePath.toFile());
            List<BoundingBox> boxes = regions.stream().map(TextRegion::box).toList();
            BufferedImage annotated = ImageAnnotator.annotate(image, boxes);
            ImageAnnotator.save(annotated, annotatedPath);
            System.out.println();
            System.out.printf("Annotated image saved to %s%n", annotatedPath);
        }
    }

    private static Path annotatedPath(Path original) {
        String filename = original.getFileName().toString();
        int dot = filename.lastIndexOf('.');
        String name = filename.substring(0, dot);
        String ext = filename.substring(dot);
        return original.getParent().resolve(name + "_annotated" + ext);
    }

    private static void printRegions(List<TextRegion> regions) {
        if (regions.isEmpty()) {
            System.out.println("  No text regions detected.");
            return;
        }
        for (int i = 0; i < regions.size(); i++) {
            TextRegion r = regions.get(i);
            System.out.printf("  %d. [%.1f, %.1f, %.1f, %.1f] (confidence=%.4f)%n",
                    i + 1,
                    r.box().x1(), r.box().y1(), r.box().x2(), r.box().y2(),
                    r.confidence());
        }
    }
}
