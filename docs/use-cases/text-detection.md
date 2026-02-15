# Text Detection

Detect text regions in images using the CRAFT model. Returns bounding boxes around detected text — useful as the first stage of an OCR pipeline.

## Quick example

```java
try (var detector = CraftTextDetector.builder().build()) {
    List<TextRegion> regions = detector.detect(Path.of("document.jpg"));
}
```

## Full example

```java
import io.github.inference4j.vision.CraftTextDetector;
import io.github.inference4j.vision.TextRegion;
import java.nio.file.Path;

public class TextDetection {
    public static void main(String[] args) {
        try (var detector = CraftTextDetector.builder().build()) {
            List<TextRegion> regions = detector.detect(Path.of("document.jpg"));

            for (TextRegion r : regions) {
                System.out.printf("[%.1f, %.1f, %.1f, %.1f] (confidence=%.2f)%n",
                    r.box().x1(), r.box().y1(),
                    r.box().x2(), r.box().y2(),
                    r.confidence());
            }
        }
    }
}
```

## Custom thresholds

The default thresholds (0.7, 0.4) are tuned for clean document scans. For real-world images (photos, screenshots), use lower thresholds:

```java
try (var detector = CraftTextDetector.builder().build()) {
    List<TextRegion> regions = detector.detect(
        Path.of("photo.jpg"), 0.4f, 0.3f);
}
```

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/craft-mlt-25k` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.inputName(String)` | `String` | auto-detected | Input tensor name |
| `.targetSize(int)` | `int` | `1280` | Long-side resize target |
| `.textThreshold(float)` | `float` | `0.7` | Region score threshold |
| `.lowTextThreshold(float)` | `float` | `0.4` | Affinity score threshold |
| `.minComponentArea(int)` | `int` | `10` | Minimum region pixel area |

## Result type

`TextRegion` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `box()` | `BoundingBox` | Bounding box around the detected text |
| `confidence()` | `float` | Detection confidence (0.0 to 1.0) |

## How CRAFT works

CRAFT (Character Region Awareness for Text detection) produces two pixel-level heatmaps:

1. **Region score** — highlights character centers
2. **Affinity score** — highlights spacing between characters (links characters into words)

Connected components on the thresholded heatmaps produce text region bounding boxes. Unlike YOLO-style detectors, CRAFT doesn't use NMS — the connected component approach naturally handles overlapping text.

## Threshold tuning

| Scenario | `textThreshold` | `lowTextThreshold` |
|----------|-----------------|---------------------|
| Clean document scans | `0.7` (default) | `0.4` (default) |
| Real-world photos | `0.4` | `0.3` |
| Sparse text (signs, labels) | `0.3` | `0.2` |

Lower thresholds detect more text but may produce false positives. Higher thresholds are more precise but may miss faint or small text.

## Hardware acceleration

CRAFT benefits significantly from hardware acceleration:

| Backend | Latency | Speedup |
|---------|---------|---------|
| CPU | 831 ms | — |
| CoreML | 153 ms | **5.4x** |

```java
try (var detector = CraftTextDetector.builder()
        .sessionOptions(opts -> opts.addCoreML())
        .build()) {
    detector.detect(Path.of("document.jpg"));
}
```

## Tips

- Both `Path` and `BufferedImage` inputs are supported.
- Images are automatically resized (preserving aspect ratio) to fit within `targetSize`. Dimensions are rounded to multiples of 32 (VGG16 backbone requirement).
- CRAFT detects text *locations*, not text *content*. Pair it with a text recognition model (like TrOCR) for full OCR.
- Coordinates are in the original image's pixel space.
