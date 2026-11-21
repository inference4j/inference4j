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

import io.github.inference4j.vision.Detection;
import io.github.inference4j.vision.DetrDetector;
import io.github.inference4j.vision.ObjectDetector;
import io.github.inference4j.vision.Yolo26Detector;

import java.nio.file.Path;
import java.util.List;

/**
 * Demonstrates object detection with D-FINE small (Apache-2.0, the default) and swapping in YOLO26n through the same
 * {@link ObjectDetector} interface. YOLO26n's weights are AGPL-3.0.
 *
 * Requires ONNX models and a sample image — see inference4j-examples/README.md for setup.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ObjectDetectionExample
 */
public class ObjectDetectionExample {

    public static void main(String[] args) throws Exception {
        Path imagePath = Path.of(ObjectDetectionExample.class.getResource("/fixtures/sample.jpg").toURI());

        // --- D-FINE small (Apache-2.0, NMS-free) ---
        System.out.println("=== D-FINE small Object Detection ===");
        try (ObjectDetector detector = DetrDetector.builder().build()) {
            System.out.println("Detections (default threshold=0.5):");
            printDetections(detector.detect(imagePath));

            System.out.println();

            System.out.println("Detections (confidence=0.8):");
            printDetections(detector.detect(imagePath, 0.8f, 0f));
        }

        System.out.println();

        // --- YOLO26n (AGPL-3.0 weights), same interface ---
        System.out.println("=== YOLO26n Object Detection (AGPL-3.0 weights) ===");
        try (ObjectDetector detector = Yolo26Detector.builder().build()) {
            System.out.println("Detections (default threshold=0.5):");
            printDetections(detector.detect(imagePath));
        }
    }

    private static void printDetections(List<Detection> detections) {
        if (detections.isEmpty()) {
            System.out.println("  No objects detected.");
            return;
        }
        for (int i = 0; i < detections.size(); i++) {
            Detection d = detections.get(i);
            System.out.printf("  %d. %s (confidence=%.4f) [%.1f, %.1f, %.1f, %.1f]%n",
                    i + 1, d.label(), d.confidence(),
                    d.box().x1(), d.box().y1(), d.box().x2(), d.box().y2());
        }
    }
}
