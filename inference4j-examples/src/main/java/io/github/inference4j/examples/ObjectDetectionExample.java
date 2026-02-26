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
import io.github.inference4j.vision.Yolo26Detector;
import io.github.inference4j.vision.YoloV8Detector;

import java.nio.file.Path;
import java.util.List;

/**
 * Demonstrates object detection with YOLOv8n and YOLO26n.
 *
 * Requires ONNX models and a sample image — see inference4j-examples/README.md for setup.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ObjectDetectionExample
 */
public class ObjectDetectionExample {

    public static void main(String[] args) throws Exception {
        Path imagePath = Path.of(ObjectDetectionExample.class.getResource("/fixtures/sample.jpg").toURI());

        // --- YOLOv8n (NMS-based) ---
        System.out.println("=== YOLOv8n Object Detection ===");
        try (YoloV8Detector yolo = YoloV8Detector.builder().build()) {
            System.out.println("YOLOv8n loaded successfully.");
            System.out.println();

            List<Detection> detections = yolo.detect(imagePath);

            System.out.println("Detections (default thresholds):");
            printDetections(detections);

            System.out.println();

            // --- Custom thresholds ---
            System.out.println("Detections (confidence=0.5, IoU=0.4):");
            List<Detection> filtered = yolo.detect(imagePath, 0.5f, 0.4f);
            printDetections(filtered);
        }

        System.out.println();

        // --- YOLO26n (NMS-free) ---
        System.out.println("=== YOLO26n Object Detection (NMS-free) ===");
        try (Yolo26Detector yolo = Yolo26Detector.builder().build()) {
            System.out.println("YOLO26n loaded successfully.");
            System.out.println();

            List<Detection> detections = yolo.detect(imagePath);

            System.out.println("Detections (default threshold=0.5):");
            printDetections(detections);

            System.out.println();

            System.out.println("Detections (confidence=0.7):");
            List<Detection> filtered = yolo.detect(imagePath, 0.7f, 0f);
            printDetections(filtered);
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
