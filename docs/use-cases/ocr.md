# OCR (Text Recognition)

Find the lines of text in an image and read them, locally, with PaddleOCR's PP-OCRv6 models. Each line comes back
with its text, a confidence and its outline in the image, in reading order.

## Quick example

```java
try (var ocr = OcrPipeline.builder().build()) {
    for (TextLine line : ocr.read(Path.of("receipt.jpg"))) {
        System.out.println(line.text());
    }
}
```

## Full example

```java
import io.github.inference4j.vision.OcrPipeline;
import io.github.inference4j.vision.TextLine;
import java.nio.file.Path;
import java.util.List;

public class ReadReceipt {
    public static void main(String[] args) {
        try (var ocr = OcrPipeline.builder()
                .minConfidence(0.5f)          // drop lines the recognizer is unsure about
                .build()) {
            List<TextLine> lines = ocr.read(Path.of("receipt.jpg"));
            for (TextLine line : lines) {
                System.out.printf("%-40s %.2f  at %s%n",
                    line.text(), line.confidence(), line.region().box());
            }
        }
    }
}
```

## How it works

Recognition models read **one line at a time**: they resize their input to a fixed height (48 pixels), so a whole
page given to them directly produces nothing useful. `OcrPipeline` therefore runs two models:

1. **Detect**: `PaddleTextDetector` finds every line of text, including rotated ones, as a four-corner outline.
2. **Crop**: `QuadCrop` cuts each outline out of the image as an upright rectangle (a perspective transform); a crop
   much taller than wide, such as vertical text, is turned 90°.
3. **Recognize**: `PaddleTextRecognizer` reads each crop and returns its text and confidence.

Lines are returned top to bottom, then left to right.

## Models

| Preset | Detector | Recognizer | Total size | Notes |
|--------|----------|------------|------------|-------|
| `small()` (default) | `inference4j/pp-ocrv6-small-det` | `inference4j/pp-ocrv6-small-rec` | ~31 MB | Dictionary of 18,708 characters: Latin scripts, Chinese, Japanese, symbols |
| `tiny()` | `inference4j/pp-ocrv6-tiny-det` | `inference4j/pp-ocrv6-tiny-rec` | ~6 MB | Dictionary of 6,904 characters; a little less accurate |

Both are mirrors of PaddlePaddle's official ONNX exports (Apache-2.0). To use the tiny models:

```java
try (var ocr = OcrPipeline.builder()
        .detector(PaddleTextDetector.tiny().build())
        .recognizer(PaddleTextRecognizer.tiny().build())
        .build()) {
    ocr.read(image);
}
```

## Builder options

### `OcrPipeline.builder()`

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.detector(TextDetector)` | `TextDetector` | `PaddleTextDetector.small()` | Finds the text regions |
| `.recognizer(TextRecognizer)` | `TextRecognizer` | `PaddleTextRecognizer.small()` | Reads each region |
| `.minConfidence(float)` | `float` | `0` | Lines read with a lower confidence are left out; empty lines always are |

Closing the pipeline closes both stages.

### `PaddleTextDetector`

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/pp-ocrv6-small-det` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.threshold(float)` | `float` | `0.3` | Probability above which a pixel of the model's map counts as text |
| `.boxThreshold(float)` | `float` | `0.6` | Minimum mean probability inside a line for it to be kept |
| `.unclipRatio(float)` | `float` | `1.5` | How far lines are expanded beyond the model's shrunken text core |
| `.maxCandidates(int)` | `int` | `3000` | Maximum candidate areas examined per image |
| `.minSide(int)` | `int` | `64` | Images with a shorter side below this are scaled up |
| `.maxSide(int)` | `int` | `4000` | Images with a longer side above this are scaled down |

The defaults are those of the `paddleocr` Python package's OCR pipeline. Images otherwise keep their own size
(rounded to multiples of 32), so small text in large images stays readable, at the cost of time on large images.

### `PaddleTextRecognizer`

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/pp-ocrv6-small-rec` | HuggingFace model ID; the directory needs `model.onnx` and `inference.yml` |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |

## Result types

`TextLine` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `text()` | `String` | The recognized text |
| `confidence()` | `float` | The recognizer's confidence (0.0 to 1.0): the mean probability of the characters read |
| `region()` | `TextRegion` | Where the line is: `box()` (axis-aligned) and `polygon()` (four corners, clockwise from the top-left) |

`TextRecognizer.recognize(BufferedImage)` returns a `RecognizedText(text, confidence)` for one line image.

## Using the stages separately

The detector and the recognizer are ordinary tasks, so each can be used alone, for example to find where text is
without reading it, or to read crops you already have:

```java
try (var detector = PaddleTextDetector.small().build();
     var recognizer = PaddleTextRecognizer.small().build()) {
    for (TextRegion region : detector.detect(page)) {
        RecognizedText text = recognizer.recognize(QuadCrop.crop(page, region.polygon()));
    }
}
```

`CraftTextDetector` also fits `OcrPipeline.builder().detector(...)`. CRAFT finds words or word groups rather than
whole lines, so each word is read separately; its default thresholds can miss clean, digitally rendered text (see
[Text Detection](text-detection.md)).

## Measured accuracy

On five test pages (a document with a rotated line, a slightly rotated JPEG receipt, colored signage with a tilted
line, a small-font screenshot, and a blurred, noisy scan: 31 lines in total), with known ground truth:

| Preset | Character accuracy | Word accuracy | Median time per page |
|--------|--------------------|---------------|----------------------|
| `small()` | 100% | 100% | 133 ms |
| `tiny()` | 99.8% | 98.6% | 84 ms |

Times are on a laptop CPU with four ONNX Runtime threads, for pages of 900 × 420 to 1200 × 820 pixels.

inference4j reproduces PaddleOCR's own processing step by step, including OpenCV's integer and single-precision
rounding, which decides where box edges fall:

- **Against PaddleOCR's Python processing on the same ONNX models:** identical text on all 31 lines, box corners
  identical or within one pixel, confidences within 0.0004.
- **Against the `paddleocr` package:** identical text on all 31 lines. That package runs Paddle's own inference
  engine on Paddle-format weights, so its boxes differ by up to a few pixels and its confidences by a few hundredths.

## Limits

- **No orientation correction.** Upside-down lines and rotated pages are not turned upright; PaddleOCR's optional
  orientation classifiers are not included.
- **Printed text.** Handwriting is better served by TrOCR ([#128](https://github.com/inference4j/inference4j/issues/128)).
- **Lines, not layout.** Columns, tables and paragraphs are not reconstructed; lines come back in reading order.
