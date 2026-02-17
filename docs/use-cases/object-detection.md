# Object Detection

Detect and locate objects in images using YOLO models. inference4j supports both NMS-based (YOLOv8, YOLO11) and NMS-free (YOLO26) architectures.

## Quick example

```java
try (var detector = YoloV8Detector.builder().build()) {
    List<Detection> detections = detector.detect(Path.of("street.jpg"));
    // [Detection[label=car, confidence=0.94, box=BoundingBox[...]], ...]
}
```

## Full example

```java
import io.github.inference4j.vision.YoloV8Detector;
import io.github.inference4j.vision.Detection;
import java.nio.file.Path;

public class ObjectDetection {
    public static void main(String[] args) {
        try (var detector = YoloV8Detector.builder().build()) {
            List<Detection> detections = detector.detect(Path.of("street.jpg"));

            for (Detection d : detections) {
                System.out.printf("%-12s (%.0f%%) at [%.0f, %.0f, %.0f, %.0f]%n",
                    d.label(), d.confidence() * 100,
                    d.box().x1(), d.box().y1(), d.box().x2(), d.box().y2());
            }
        }
    }
}
```

<figure markdown="span">
  ![Screenshot from showcase app](../assets/images/object-detection.png)
  <figcaption>Screenshot from showcase app</figcaption>
</figure>

## Available models

### YOLOv8 / YOLO11

NMS-based detection. The `YoloV8Detector` is compatible with both YOLOv8 and YOLO11 models (same output layout).

```java
try (var detector = YoloV8Detector.builder().build()) {
    detector.detect(Path.of("image.jpg"));
}
```

### YOLO26

NMS-free detection. The architecture outputs 300 deduplicated proposals directly — no post-processing NMS needed.

```java
try (var detector = Yolo26Detector.builder().build()) {
    detector.detect(Path.of("image.jpg"));
}
```

!!! note
    YOLO26 accepts an `iouThreshold` parameter for API compatibility, but it is ignored — the architecture handles deduplication internally.

## YoloV8 builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/yolov8n` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.labels(Labels)` | `Labels` | COCO (80 classes) | Detection class labels |
| `.inputName(String)` | `String` | auto-detected | Input tensor name |
| `.inputSize(int)` | `int` | `640` | Model input resolution |
| `.confidenceThreshold(float)` | `float` | `0.25` | Minimum detection confidence |
| `.iouThreshold(float)` | `float` | `0.45` | NMS IoU threshold |

## Yolo26 builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/yolo26n` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.labels(Labels)` | `Labels` | COCO (80 classes) | Detection class labels |
| `.inputName(String)` | `String` | auto-detected | Input tensor name |
| `.inputSize(int)` | `int` | `640` | Model input resolution |
| `.confidenceThreshold(float)` | `float` | `0.5` | Minimum detection confidence |

## Result type

`Detection` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `label()` | `String` | Class label (e.g., `car`, `person`) |
| `classIndex()` | `int` | Class index in the label set |
| `confidence()` | `float` | Detection confidence (0.0 to 1.0) |
| `box()` | `BoundingBox` | Bounding box coordinates |

`BoundingBox` provides:

| Field | Type | Description |
|-------|------|-------------|
| `x1()` | `float` | Left coordinate |
| `y1()` | `float` | Top coordinate |
| `x2()` | `float` | Right coordinate |
| `y2()` | `float` | Bottom coordinate |

Coordinates are in the original image's pixel space (not normalized).

## Custom thresholds

Override thresholds per-call:

```java
try (var detector = YoloV8Detector.builder().build()) {
    // Stricter confidence, tighter NMS
    List<Detection> detections = detector.detect(
        Path.of("image.jpg"), 0.5f, 0.3f);
}
```

## Tips

- **YOLOv8 vs YOLO26**: YOLOv8 is the safer default — widely tested and well-understood. YOLO26 is NMS-free (potentially faster) but newer.
- Both detectors accept `Path` or `BufferedImage` as input.
- Default labels are COCO (80 classes). Override with `.labels(Labels.of(yourLabels))` for custom-trained models.
- YOLOv8 uses letterbox preprocessing (preserves aspect ratio, pads with gray). YOLO26 uses direct resize.
