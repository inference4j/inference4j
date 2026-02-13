# inference4j

inference4j is a modern, type-safe, and ergonomic AI inference library for Java, built on top of the ONNX Runtime. It aims to make integrating AI models into Java applications as simple as any other library, without sacrificing performance or type safety.

## Quick Start

```java
// Text embeddings
try (SentenceTransformer model = SentenceTransformer.fromPretrained("models/all-MiniLM-L6-v2")) {
    float[] embedding = model.encode("Hello, world!");
}

// Image classification
try (ResNet model = ResNet.fromPretrained("models/resnet50")) {
    List<Classification> results = model.classify(Path.of("cat.jpg"));
}

// Object detection
try (YoloV8 model = YoloV8.fromPretrained("models/yolov8n")) {
    List<Detection> detections = model.detect(Path.of("street.jpg"));
}

// Speech-to-text
try (Wav2Vec2 model = Wav2Vec2.fromPretrained("models/wav2vec2-base-960h")) {
    Transcription result = model.transcribe(Path.of("audio.wav"));
    System.out.println(result.text());
}
```

## Supported Models

| Domain | Model | Wrapper | Description |
|--------|-------|---------|-------------|
| **Text** | all-MiniLM, all-mpnet, BERT | `SentenceTransformer` | Sentence embeddings with configurable pooling |
| **Vision** | ResNet | `ResNet` | Image classification (ImageNet) |
| **Vision** | EfficientNet | `EfficientNet` | Image classification (ImageNet) |
| **Vision** | YOLOv8, YOLO11 | `YoloV8` | Object detection with NMS |
| **Vision** | YOLO26 | `Yolo26` | NMS-free object detection |
| **Audio** | Wav2Vec2-CTC | `Wav2Vec2` | Speech-to-text (single-pass, non-autoregressive) |

## Project Structure

- `inference4j-core`: Core abstractions — `InferenceSession`, `Tensor`, `ModelSource`, `MathOps`.
- `inference4j-preprocessing`: Data preparation — tokenizers, image transforms, audio processing.
- `inference4j-models`: Handcrafted model wrappers with domain-specific APIs.
- `inference4j-runtime`: Operational layer — model registry, metrics, A/B testing.
- `inference4j-examples`: Runnable examples ([see README](inference4j-examples/README.md)).

## Build

Requires **Java 21** and **Gradle 9.2.1**.

```bash
./gradlew build          # Build all modules
./gradlew test           # Run all tests
```

## Documentation

- [Vision](docs/vision.md) - Why we are building inference4j and our core principles.
- [Architecture](docs/architecture.md) - How the project is structured.
- [API Design](docs/api-design.md) - Examples of the intended developer experience.
- [Roadmap](docs/roadmap.md) - Our plan for development.
- [Initial Brainstorm](docs/brainstorm-initial.md) - The original discussion that started the project.
