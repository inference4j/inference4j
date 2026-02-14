# inference4j

[![CI](https://github.com/inference4j/inference4j/actions/workflows/ci.yml/badge.svg)](https://github.com/inference4j/inference4j/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/inference4j/inference4j/graph/badge.svg)](https://codecov.io/gh/inference4j/inference4j)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**Run AI models in Java. Three lines of code, zero setup.**

> **Note:** inference4j is under active development (0.x). APIs may change. A full user guide and wiki will follow — for now, this README and the [examples](inference4j-examples/README.md) are the best way to get started.

## What can you do with inference4j?

### Sentiment Analysis

```java
try (var model = DistilBertClassifier.builder().build()) {
    System.out.println(model.classify("This movie was fantastic!"));
    // [TextClassification[label=POSITIVE, confidence=0.9998]]
}
```

### Text Embeddings & Semantic Search

```java
try (var model = SentenceTransformer.builder()
        .modelId("inference4j/all-MiniLM-L6-v2").build()) {
    float[] embedding = model.encode("Hello, world!");
}
```

### Image Classification

```java
try (var model = ResNet.builder().build()) {
    List<Classification> results = model.classify(Path.of("cat.jpg"));
    // [Classification[label=tabby cat, confidence=0.87], ...]
}
```

### Object Detection

```java
try (var model = YoloV8.builder().build()) {
    List<Detection> detections = model.detect(Path.of("street.jpg"));
    // [Detection[label=car, confidence=0.94, box=BoundingBox[...]], ...]
}
```

### Speech-to-Text

```java
try (var model = Wav2Vec2.builder().build()) {
    System.out.println(model.transcribe(Path.of("audio.wav")).text());
}
```

### Voice Activity Detection

```java
try (var vad = SileroVAD.builder().build()) {
    List<VoiceSegment> segments = vad.detect(Path.of("meeting.wav"));
    // [VoiceSegment[start=0.50, end=3.20], VoiceSegment[start=5.10, end=8.75]]
}
```

### Text Detection

```java
try (var craft = Craft.builder().build()) {
    List<TextRegion> regions = craft.detect(Path.of("document.jpg"));
}
```

### Search Reranking

```java
try (var reranker = MiniLMReranker.builder().build()) {
    float score = reranker.score("What is Java?", "Java is a programming language.");
}
```

## What you don't have to do

- **No tokenization** — WordPiece tokenizers are built in and handled automatically
- **No tensor handling** — pass a `String`, `BufferedImage`, or `Path`; get Java objects back
- **No ONNX session setup** — `builder().build()` handles everything
- **No model downloads** — auto-downloaded from HuggingFace and cached on first use
- **No Python sidecar** — pure Java, runs anywhere Java runs

## vs raw ONNX Runtime

<table>
<tr>
<th>Without inference4j</th>
<th>With inference4j</th>
</tr>
<tr>
<td>

```java
OrtEnvironment env = OrtEnvironment.getEnvironment();
OrtSession session = env.createSession("resnet50.onnx");

BufferedImage img = ImageIO.read(new File("cat.jpg"));
BufferedImage resized = resize(img, 224, 224);
float[] pixels = new float[3 * 224 * 224];
for (int c = 0; c < 3; c++)
  for (int y = 0; y < 224; y++)
    for (int x = 0; x < 224; x++) {
      int rgb = resized.getRGB(x, y);
      float val = ((rgb >> (16 - c * 8)) & 0xFF) / 255f;
      pixels[c * 224 * 224 + y * 224 + x] =
        (val - MEAN[c]) / STD[c];
    }

OnnxTensor tensor = OnnxTensor.createTensor(env,
    FloatBuffer.wrap(pixels), new long[]{1, 3, 224, 224});
OrtSession.Result result = session.run(
    Map.of("data", tensor));
float[] logits = ((float[][]) result.get(0)
    .getValue())[0];
float[] probs = softmax(logits);
int bestIdx = argmax(probs);
String label = LABELS[bestIdx];
// ~30 lines, manual everything
```

</td>
<td>

```java
try (var model = ResNet.builder().build()) {
    var results = model.classify(
        Path.of("cat.jpg")
    );
    // done.
}

// 3 lines.
// Auto-downloads model.
// Handles preprocessing.
// Returns Java objects.
```

</td>
</tr>
</table>

## Why inference4j?

Java has great tools for building AI-powered applications. [Spring AI](https://spring.io/projects/spring-ai) provides an excellent abstraction layer for LLM orchestration. [DJL](https://djl.ai/) offers engine-agnostic model training and inference. [LangChain4j](https://docs.langchain4j.dev/) simplifies LLM-powered workflows.

**inference4j doesn't compete with any of them.** It fills a different gap.

When you need to run a specific ONNX model — an embedding model, an object detector, a speech-to-text model — you currently face a choice: drop down to the raw ONNX Runtime Java bindings and deal with `Map<String, OnnxTensor>` manually, or pull in a heavyweight framework that does far more than you need.

inference4j sits in the sweet spot:

- **3-line integration** for popular models — `builder().build()`, call a method, get Java objects back
- **Standard Java types** in, standard Java types out — no tensor abstractions leak into your code
- **Inference only** — optimized for production serving, not training
- **Lightweight** — each wrapper is a thin layer over ONNX Runtime, not a framework
- **Complements the ecosystem** — use inference4j to run your embedding model, Spring AI to orchestrate your LLM chain, both in the same application

We believe the Java AI ecosystem is stronger when tools do one thing well. inference4j does local model inference, and tries to do it really well.

## Supported Tasks

| Task | Models | Wrapper |
|------|--------|---------|
| **Sentiment Analysis** | DistilBERT, BERT | `DistilBertClassifier` |
| **Text Embeddings** | all-MiniLM, all-mpnet, BERT | `SentenceTransformer` |
| **Search Reranking** | ms-marco-MiniLM | `MiniLMReranker` |
| **Image Classification** | ResNet, EfficientNet | `ResNet`, `EfficientNet` |
| **Object Detection** | YOLOv8, YOLO11, YOLO26 | `YoloV8`, `Yolo26` |
| **Text Detection** | CRAFT | `Craft` |
| **Speech-to-Text** | Wav2Vec2-CTC | `Wav2Vec2` |
| **Voice Activity Detection** | Silero VAD | `SileroVAD` |

> **Auto-download:** All supported models are hosted under the [`inference4j`](https://huggingface.co/inference4j) HuggingFace organization. Models are automatically downloaded and cached on first use — no manual setup required. Cache location defaults to `~/.cache/inference4j/` and can be customized via `INFERENCE4J_CACHE_DIR` or `-Dinference4j.cache.dir`.

## Roadmap

- **OCR Pipeline** — CRAFT text detection + TrOCR recognition + embedding-based error correction against domain dictionaries
- **Pipeline API** — compose models into multi-stage workflows with per-stage timing and intermediate hooks
- **CLIP** — image-text similarity for visual search and zero-shot classification
- **Spring Boot Starter** — auto-configuration, health indicators, Micrometer metrics

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
