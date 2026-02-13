# inference4j

[![CI](https://github.com/inference4j/inference4j/actions/workflows/ci.yml/badge.svg)](https://github.com/inference4j/inference4j/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/inference4j/inference4j/graph/badge.svg)](https://codecov.io/gh/inference4j/inference4j)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Inference-only AI for Java. Simple APIs, standard types, no PhD required.**

> **Note:** inference4j is under active development. APIs may change as we stabilize. A full user guide and wiki will follow — for now, this README and the [examples](inference4j-examples/README.md) are the best way to get started.

inference4j gives Java developers ergonomic, type-safe wrappers for running AI models on the ONNX Runtime. Pass in a `String`, a `BufferedImage`, or a `Path` to a WAV file — get back embeddings, classifications, detections, or transcriptions. No tensor juggling, no JNI plumbing, no Python sidecar.

```java
try (Wav2Vec2 model = Wav2Vec2.fromPretrained("models/wav2vec2-base-960h")) {
    Transcription result = model.transcribe(Path.of("meeting.wav"));
    System.out.println(result.text());
}
```

## Why inference4j?

Java has great tools for building AI-powered applications. [Spring AI](https://spring.io/projects/spring-ai) provides an excellent abstraction layer for LLM orchestration. [DJL](https://djl.ai/) offers engine-agnostic model training and inference. [LangChain4j](https://docs.langchain4j.dev/) simplifies LLM-powered workflows.

**inference4j doesn't compete with any of them.** It fills a different gap.

When you need to run a specific ONNX model — an embedding model, an object detector, a speech-to-text model — you currently face a choice: drop down to the raw ONNX Runtime Java bindings and deal with `Map<String, OnnxTensor>` manually, or pull in a heavyweight framework that does far more than you need.

inference4j sits in the sweet spot:

- **3-line integration** for popular models — `fromPretrained()`, call a method, get Java objects back
- **Standard Java types** in, standard Java types out — no tensor abstractions leak into your code
- **Inference only** — optimized for production serving, not training
- **Lightweight** — each wrapper is a thin layer over ONNX Runtime, not a framework
- **Complements the ecosystem** — use inference4j to run your embedding model, Spring AI to orchestrate your LLM chain, both in the same application

We believe the Java AI ecosystem is stronger when tools do one thing well. inference4j does local model inference, and tries to do it really well.

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

// Text classification (sentiment analysis)
try (DistilBertClassifier model = DistilBertClassifier.fromPretrained("models/distilbert-sst2")) {
    List<TextClassification> results = model.classify("This movie was fantastic!");
    System.out.println(results.get(0).label()); // "POSITIVE"
}

// Cross-encoder reranking
try (MiniLMReranker reranker = MiniLMReranker.fromPretrained("models/ms-marco-MiniLM")) {
    float score = reranker.score("What is Java?", "Java is a programming language.");
}

// Speech-to-text
try (Wav2Vec2 model = Wav2Vec2.fromPretrained("models/wav2vec2-base-960h")) {
    Transcription result = model.transcribe(Path.of("audio.wav"));
    System.out.println(result.text());
}

// Voice activity detection
try (SileroVAD vad = SileroVAD.fromPretrained("models/silero-vad")) {
    List<VoiceSegment> segments = vad.detect(Path.of("meeting.wav"));
    for (VoiceSegment segment : segments) {
        System.out.printf("Speech: %.2fs - %.2fs%n", segment.start(), segment.end());
    }
}

// Text detection
try (Craft craft = Craft.fromPretrained("models/craft")) {
    List<TextRegion> regions = craft.detect(Path.of("document.jpg"));
    for (TextRegion r : regions) {
        System.out.printf("Text at [%.0f, %.0f, %.0f, %.0f]%n",
            r.box().x1(), r.box().y1(), r.box().x2(), r.box().y2());
    }
}
```

## Supported Models

| Domain | Model | Wrapper | Description |
|--------|-------|---------|-------------|
| **Text** | all-MiniLM, all-mpnet, BERT | `SentenceTransformer` | Sentence embeddings with configurable pooling |
| **Text** | DistilBERT, BERT (classification) | `DistilBertClassifier` | Text classification — sentiment, moderation, intent detection |
| **Text** | ms-marco-MiniLM (cross-encoder) | `MiniLMReranker` | Query-document relevance scoring for search reranking |
| **Vision** | ResNet | `ResNet` | Image classification (ImageNet) |
| **Vision** | EfficientNet | `EfficientNet` | Image classification (ImageNet) |
| **Vision** | YOLOv8, YOLO11 | `YoloV8` | Object detection with NMS |
| **Vision** | YOLO26 | `Yolo26` | NMS-free object detection |
| **Audio** | Wav2Vec2-CTC | `Wav2Vec2` | Speech-to-text (single-pass, non-autoregressive) |
| **Audio** | Silero VAD | `SileroVAD` | Voice activity detection |
| **Vision** | CRAFT | `Craft` | Text detection — locates text regions in images |

> **CRAFT ONNX model:** We converted CRAFT from the [original PyTorch weights](https://github.com/clovaai/CRAFT-pytorch) (`craft_mlt_25k.pth`) to ONNX and host it at [`inference4j/craft-mlt-25k`](https://huggingface.co/inference4j/craft-mlt-25k). The conversion script is included in the repo for reproducibility.

## Vision

inference4j follows a three-tier API strategy:

1. **Handcrafted wrappers** — curated, ergonomic APIs for the most popular models (what you see above)
2. **Code-generated wrappers** — a Gradle plugin that reads `.onnx` files and generates type-safe Java classes for any model
3. **Low-level core** — direct `InferenceSession` and `Tensor` access when you need full control

On our roadmap:

- **CLIP** — image-text similarity for visual search and zero-shot classification
- **OCR Pipeline** — text recognition (TrOCR) + embedding-based error correction against domain dictionaries (CRAFT text detection is already available)
- **Pipeline API** — compose models into multi-stage workflows with per-stage timing and intermediate hooks
- **Spring Boot Starter** — auto-configuration, health indicators, Micrometer metrics
- **HuggingFace integration** — `ModelSource` that downloads and caches models from the Hub

See the [Roadmap](ROADMAP.md) for details.

## Project Structure

| Module | Description |
|--------|-------------|
| `inference4j-core` | Low-level ONNX Runtime abstractions — `InferenceSession`, `Tensor`, `ModelSource`, `MathOps` |
| `inference4j-preprocessing` | Tokenizers, image transforms, audio processing |
| `inference4j-models` | Handcrafted model wrappers with domain-specific APIs |
| `inference4j-runtime` | Operational layer — model routing, A/B testing, Micrometer metrics |
| `inference4j-examples` | Runnable examples ([see README](inference4j-examples/README.md)) |

## Build

Requires **Java 21**.

```bash
./gradlew build          # Build all modules and run tests
./gradlew test           # Run tests only
```

## Built with Claude Code

This project was built collaboratively with [Claude Code](https://claude.ai/code).

The humans drove architecture, API design, interface contracts, and model selection. Claude Code wrote the implementation — the wrappers, the preprocessing pipelines, the math utilities, the tests, and the documentation you're reading right now.

Without Claude Code, this project would have taken weeks instead of hours. We want to embrace this new reality of software development, keep pushing forward, and — with community feedback and contributions — give something useful back to the Java ecosystem.

We think this is worth being transparent about. Agentic development is how software gets built now, and pretending otherwise helps no one. The design decisions are ours; the execution was a partnership.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[Apache License 2.0](LICENSE)
