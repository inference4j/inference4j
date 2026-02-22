# inference4j

[![CI](https://github.com/inference4j/inference4j/actions/workflows/ci.yml/badge.svg)](https://github.com/inference4j/inference4j/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/inference4j/inference4j/graph/badge.svg)](https://codecov.io/gh/inference4j/inference4j)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-inference4j.github.io-7c4dff.svg)](https://inference4j.github.io/inference4j)

**Run AI models in Java. Three lines of code, zero setup.**

inference4j is an inference-only AI library for Java built on ONNX Runtime. It provides ergonomic, type-safe APIs for running model inference **locally** — no API keys, no network calls, no third-party services. Pass a `String`, `BufferedImage`, or `Path`, get Java objects back.

> **Note:** inference4j is under active development (0.x). APIs may change. Check out the [documentation site](https://inference4j.github.io/inference4j) for the full user guide, or browse the [examples](inference4j-examples/README.md) to get started.

## What can you do with inference4j?

Want to see it in action? Check out [inference4j-showcase](https://github.com/inference4j/inference4j-showcase) — a local demo app you can run to explore every capability the library provides.

### Sentiment Analysis

```java
try (var classifier = DistilBertTextClassifier.builder().build()) {
    System.out.println(classifier.classify("This movie was fantastic!"));
    // [TextClassification[label=POSITIVE, confidence=0.9998]]
}
```

### Text Embeddings & Semantic Search

```java
try (var embedder = SentenceTransformerEmbedder.builder()
        .modelId("inference4j/all-MiniLM-L6-v2").build()) {
    float[] embedding = embedder.encode("Hello, world!");
}
```

### Image Classification

```java
try (var classifier = ResNetClassifier.builder().build()) {
    List<Classification> results = classifier.classify(Path.of("cat.jpg"));
    // [Classification[label=tabby cat, confidence=0.87], ...]
}
```

### Object Detection

```java
try (var detector = YoloV8Detector.builder().build()) {
    List<Detection> detections = detector.detect(Path.of("street.jpg"));
    // [Detection[label=car, confidence=0.94, box=BoundingBox[...]], ...]
}
```

### Speech-to-Text

```java
try (var recognizer = Wav2Vec2Recognizer.builder().build()) {
    System.out.println(recognizer.transcribe(Path.of("audio.wav")).text());
}
```

### Voice Activity Detection

```java
try (var vad = SileroVadDetector.builder().build()) {
    List<VoiceSegment> segments = vad.detect(Path.of("meeting.wav"));
    // [VoiceSegment[start=0.50, end=3.20], VoiceSegment[start=5.10, end=8.75]]
}
```

### Text Detection

```java
try (var detector = CraftTextDetector.builder().build()) {
    List<TextRegion> regions = detector.detect(Path.of("document.jpg"));
}
```

### Zero-Shot Image Classification

```java
try (var classifier = ClipClassifier.builder().build()) {
    List<Classification> results = classifier.classify(
            Path.of("photo.jpg"), List.of("cat", "dog", "bird", "car"));
    // [Classification[label=cat, confidence=0.82], ...]
}
```

### Search Reranking

```java
try (var reranker = MiniLMSearchReranker.builder().build()) {
    float score = reranker.score("What is Java?", "Java is a programming language.");
}
```

### Text Generation

```java
try (var gen = OnnxTextGenerator.qwen2()
        .maxNewTokens(50).temperature(0.8f).topK(50).build()) {
    gen.generate("Explain gravity", token -> System.out.print(token));
}
```

## Getting Started

**Requirements:** Java 17+

### Add the dependency

`inference4j-core` is the only dependency you need — it includes all model wrappers, tokenizers, and preprocessing.

**Gradle**

```groovy
implementation 'io.github.inference4j:inference4j-core:${inference4jVersion}'
```

**Maven**

```xml
<dependency>
    <groupId>io.github.inference4j</groupId>
    <artifactId>inference4j-core</artifactId>
    <version>${inference4jVersion}</version>
</dependency>
```

> **JVM flag:** ONNX Runtime requires native access. Add `--enable-native-access=ALL-UNNAMED` to your JVM arguments, or use `--enable-native-access=com.microsoft.onnxruntime` if you're on the module path.

### Run your first model

```java
import io.github.inference4j.nlp.DistilBertTextClassifier;

public class QuickStart {
    public static void main(String[] args) {
        try (var classifier = DistilBertTextClassifier.builder().build()) {
            System.out.println(classifier.classify("inference4j makes AI in Java easy!"));
            // [TextClassification[label=POSITIVE, confidence=0.9998]]
        }
    }
}
```

That's it. The model downloads automatically on first run (~260MB, cached in `~/.cache/inference4j/`). No Python, no manual downloads, no tensor wrangling.

## What you don't have to do

- **No tokenization** — WordPiece and BPE tokenizers are built in and handled automatically
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
try (var classifier = ResNetClassifier.builder().build()) {
    var results = classifier.classify(
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

## Supported Models

### NLP

| Capability | Models | API |
|---|---|---|
| Classification | DistilBERT, BERT | `TextClassifier` |
| Embeddings | all-MiniLM, all-mpnet | `TextEmbedder` |
| Reranking | ms-marco-MiniLM | `SearchReranker` |
| Text Generation | GPT-2, SmolLM2, Qwen2.5 | `TextGenerator` |

### Vision

| Capability | Models | API |
|---|---|---|
| Classification | ResNet, EfficientNet | `ImageClassifier` |
| Object Detection | YOLOv8, YOLO11, YOLO26 | `ObjectDetector` |
| Text Detection | CRAFT | `TextDetector` |

### Multimodal

| Capability | Models | API |
|---|---|---|
| Zero-Shot Classification | CLIP | `ZeroShotClassifier` |
| Image Embeddings | CLIP | `ImageEmbedder` |
| Text Embeddings | CLIP | `TextEmbedder` |

### Audio

| Capability | Models | API |
|---|---|---|
| Recognition | Wav2Vec2 | `SpeechRecognizer` |
| Voice Activity Detection | Silero VAD | `VoiceActivityDetector` |

### Generative AI (onnxruntime-genai)

| Capability | Models | API |
|---|---|---|
| Text Generation | Phi-3, DeepSeek-R1 | `TextGenerator` |
| Vision-Language | Phi-3.5 Vision | `VisionLanguageModel` |
| Speech-to-Text | Whisper | `WhisperSpeechModel` |

> **Auto-download:** All supported models are hosted under the [`inference4j`](https://huggingface.co/inference4j) HuggingFace organization. Models are automatically downloaded and cached on first use — no manual setup required. Cache location defaults to `~/.cache/inference4j/` and can be customized via `INFERENCE4J_CACHE_DIR` or `-Dinference4j.cache.dir`.

## Hardware Acceleration

inference4j supports GPU and hardware acceleration out of the box via ONNX Runtime execution providers. On macOS, CoreML is bundled in the standard dependency — just add one line:

```java
try (var classifier = ResNetClassifier.builder()
        .sessionOptions(opts -> opts.addCoreML())
        .build()) {
    classifier.classify(Path.of("cat.jpg"));
}
```

For CUDA (Linux/Windows), swap the Maven dependency from `onnxruntime` to `onnxruntime_gpu`:

```java
try (var classifier = ResNetClassifier.builder()
        .sessionOptions(opts -> opts.addCUDA(0))
        .build()) {
    classifier.classify(Path.of("cat.jpg"));
}
```

The `.sessionOptions()` API is available on every model wrapper.

### Benchmarks on Apple Silicon (M-series)

| Model | Capability | CPU | CoreML | Speedup |
|-------|------|-----|--------|---------|
| ResNet-50 | Image Classification | 37 ms | 10 ms | **3.7x** |
| CRAFT | Text Detection | 831 ms | 153 ms | **5.4x** |

> Measured with 3 warmup runs + 10 timed runs. See the benchmark examples for [ResNet](inference4j-examples/src/main/java/io/github/inference4j/examples/ResNetAccelerationBenchmarkExample.java) and [CRAFT](inference4j-examples/src/main/java/io/github/inference4j/examples/CraftAccelerationBenchmarkExample.java).

## Spring Boot

Add the starter and enable the models you need:

```groovy
implementation 'io.github.inference4j:inference4j-spring-boot-starter:${inference4jVersion}'
```

```yaml
inference4j:
  nlp:
    text-classifier:
      enabled: true
```

```java
@RestController
public class SentimentController {
    private final TextClassifier classifier;

    public SentimentController(TextClassifier classifier) {
        this.classifier = classifier;
    }

    @PostMapping("/analyze")
    public List<TextClassification> analyze(@RequestBody String text) {
        return classifier.classify(text);
    }
}
```

Every model is opt-in — nothing is downloaded until you set `enabled: true`. Beans are interface-typed, so you can swap implementations with `@ConditionalOnMissingBean`. An actuator health indicator is included out of the box. See the [full documentation](https://github.com/inference4j/inference4j/wiki) for all available properties.

## Roadmap

See the [Roadmap](https://inference4j.github.io/inference4j/roadmap/) for details and what's coming next.

## Project Structure

| Module | Description |
|--------|-------------|
| `inference4j-core` | Model wrappers, tokenizers, preprocessing, ONNX Runtime abstractions, native generation engine |
| `inference4j-genai` | Generative AI via [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) — Phi-3, DeepSeek-R1, Whisper, Phi-3.5 Vision |
| `inference4j-runtime` | Operational layer — model routing, A/B testing, Micrometer metrics |
| `inference4j-spring-boot-starter` | Spring Boot auto-configuration, health indicators |
| `inference4j-examples` | Runnable examples ([see README](inference4j-examples/README.md)) |

## Build

Requires **Java 17**.

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
