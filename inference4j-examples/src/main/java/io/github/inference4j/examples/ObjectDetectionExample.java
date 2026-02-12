package io.github.inference4j.examples;

import io.github.inference4j.vision.detection.Detection;
import io.github.inference4j.vision.detection.Yolo26;
import io.github.inference4j.vision.detection.YoloV8;

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

    public static void main(String[] args) {
        String imagePath = "inference4j-examples/images/sample.jpg";

        // --- YOLOv8n (NMS-based) ---
        System.out.println("=== YOLOv8n Object Detection ===");
        try (YoloV8 yolo = YoloV8.fromPretrained("inference4j-examples/models/yolov8n")) {
            System.out.println("YOLOv8n loaded successfully.");
            System.out.println();

            List<Detection> detections = yolo.detect(Path.of(imagePath));

            System.out.println("Detections (default thresholds):");
            printDetections(detections);

            System.out.println();

            // --- Custom thresholds ---
            System.out.println("Detections (confidence=0.5, IoU=0.4):");
            List<Detection> filtered = yolo.detect(Path.of(imagePath), 0.5f, 0.4f);
            printDetections(filtered);
        }

        System.out.println();

        // --- YOLO26n (NMS-free) ---
        System.out.println("=== YOLO26n Object Detection (NMS-free) ===");
        try (Yolo26 yolo = Yolo26.fromPretrained("inference4j-examples/models/yolo26n")) {
            System.out.println("YOLO26n loaded successfully.");
            System.out.println();

            List<Detection> detections = yolo.detect(Path.of(imagePath));

            System.out.println("Detections (default threshold=0.5):");
            printDetections(detections);

            System.out.println();

            System.out.println("Detections (confidence=0.7):");
            List<Detection> filtered = yolo.detect(Path.of(imagePath), 0.7f, 0f);
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
