# Image Classification

Classify images using ResNet or EfficientNet models. Pass an image path or `BufferedImage`, get back labeled predictions with confidence scores.

## Quick example

```java
try (var classifier = ResNetClassifier.builder().build()) {
    List<Classification> results = classifier.classify(Path.of("cat.jpg"));
    // [Classification[label=tabby cat, confidence=0.87], ...]
}
```

## Full example

```java
import io.github.inference4j.vision.ResNetClassifier;
import io.github.inference4j.vision.Classification;
import java.nio.file.Path;

public class ImageClassification {
    public static void main(String[] args) {
        try (var classifier = ResNetClassifier.builder().build()) {
            List<Classification> results = classifier.classify(Path.of("cat.jpg"), 5);

            for (Classification c : results) {
                System.out.printf("%-30s %.2f%%%n", c.label(), c.confidence() * 100);
            }
        }
    }
}
```

## Available models

### ResNet

The default choice for image classification. Uses ImageNet normalization and softmax output.

```java
try (var classifier = ResNetClassifier.builder().build()) {
    classifier.classify(Path.of("image.jpg"));
}
```

### EfficientNet

An alternative architecture with built-in softmax. Good for models exported from TensorFlow.

```java
try (var classifier = EfficientNetClassifier.builder().build()) {
    classifier.classify(Path.of("image.jpg"));
}
```

!!! note
    EfficientNet-Lite4 (the default) has softmax built into the model. If you use a PyTorch-exported EfficientNet that outputs raw logits, override with `.outputOperator(OutputOperator.softmax())`.

## ResNet builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/resnet50-v1-7` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.labels(Labels)` | `Labels` | auto-loaded from `labels.txt` | Classification labels |
| `.outputOperator(OutputOperator)` | `OutputOperator` | `softmax()` | Output activation function |
| `.defaultTopK(int)` | `int` | `5` | Default number of top predictions |

## EfficientNet builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/efficientnet-lite4` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.labels(Labels)` | `Labels` | auto-loaded from `labels.txt` | Classification labels |
| `.outputOperator(OutputOperator)` | `OutputOperator` | `identity()` | Output activation (softmax built-in) |
| `.defaultTopK(int)` | `int` | `5` | Default number of top predictions |

## Result type

`Classification` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `label()` | `String` | ImageNet class label (e.g., `tabby cat`) |
| `index()` | `int` | Class index (0–999 for ImageNet) |
| `confidence()` | `float` | Confidence score (0.0 to 1.0) |

## Hardware acceleration

Image classification benefits significantly from hardware acceleration:

| Model | CPU | CoreML | Speedup |
|-------|-----|--------|---------|
| ResNet-50 | 37 ms | 10 ms | **3.7x** |

```java
try (var classifier = ResNetClassifier.builder()
        .sessionOptions(opts -> opts.addCoreML())
        .build()) {
    classifier.classify(Path.of("cat.jpg"));
}
```

See the [Hardware Acceleration guide](../guides/hardware-acceleration.md) for details.

## Tips

- Use `classify(image, topK)` to control how many predictions are returned.
- Both classifiers accept `Path` or `BufferedImage` as input.
- Input images are automatically resized and normalized — no manual preprocessing needed.
- Input size is auto-detected from the model (ResNet: 224x224, EfficientNet-Lite4: 280x280).
