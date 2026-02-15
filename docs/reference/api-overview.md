# API Overview

## Package structure

### inference4j-core

| Package | Contents |
|---------|----------|
| `io.github.inference4j` | Core contracts: `InferenceTask`, `Classifier`, `Detector`, `AbstractInferenceTask`, `Tensor`, `TensorType`, `InferenceSession`, `InferenceContext` |
| `io.github.inference4j.session` | Session config: `SessionConfigurer`, `SessionOptions` |
| `io.github.inference4j.model` | Model resolution: `ModelSource`, `HuggingFaceModelSource`, `LocalModelSource` |
| `io.github.inference4j.processing` | Pre/post-processing: `Preprocessor`, `Postprocessor`, `OutputOperator`, `MathOps` |
| `io.github.inference4j.exception` | Custom exceptions: `ModelLoadException`, `InferenceException` |

### inference4j-preprocessing

| Package | Contents |
|---------|----------|
| `io.github.inference4j.tokenizer` | `Tokenizer`, `EncodedInput`, `WordPieceTokenizer`, `BpeTokenizer` |
| `io.github.inference4j.text` | `ModelConfig` (HuggingFace config.json parser) |
| `io.github.inference4j.image` | Image transforms pipeline |

### inference4j-tasks

| Package | Contents |
|---------|----------|
| `io.github.inference4j.vision` | `ResNetClassifier`, `EfficientNetClassifier`, `YoloV8Detector`, `Yolo26Detector`, `CraftTextDetector`, `ImageEmbedder` |
| `io.github.inference4j.audio` | `Wav2Vec2Recognizer`, `SileroVadDetector` |
| `io.github.inference4j.nlp` | `DistilBertTextClassifier`, `SentenceTransformerEmbedder`, `MiniLMSearchReranker` |
| `io.github.inference4j.multimodal` | `ClipImageEncoder`, `ClipTextEncoder` |

### inference4j-runtime

| Package | Contents |
|---------|----------|
| `io.github.inference4j.routing` | `ModelRouter`, `Route`, `RoutingStrategy`, `WeightedRoutingStrategy`, `RouterMetrics` |

### inference4j-spring-boot-starter

| Package | Contents |
|---------|----------|
| `io.github.inference4j.autoconfigure` | Auto-configuration, health indicators, `Inference4jProperties` |

## Interface hierarchy

```
InferenceTask<I, O>                     // run(I) → O, extends AutoCloseable
├── Classifier<I, C>                    // classify(I) → List<C>
│   ├── ImageClassifier                 // classify(BufferedImage/Path) → List<Classification>
│   └── TextClassifier                  // classify(String) → List<TextClassification>
├── Detector<I, D>                      // detect(I) → List<D>
│   ├── ObjectDetector                  // detect(BufferedImage/Path) → List<Detection>
│   ├── TextDetector                    // detect(BufferedImage/Path) → List<TextRegion>
│   └── VoiceActivityDetector           // detect(Path/float[]) → List<VoiceSegment>
├── TextEmbedder                        // encode(String) → float[]
├── ImageEmbedder                       // encode(BufferedImage/Path) → float[]
├── SearchReranker                      // score(String, String) → float
└── SpeechRecognizer                    // transcribe(Path) → Transcription
```

## AbstractInferenceTask

Most task wrappers extend `AbstractInferenceTask<I, O>`, which enforces a `final run()` method with three stages:

1. **Preprocess**: `I → Map<String, Tensor>` (via `Preprocessor`)
2. **Infer**: `Map<String, Tensor> → Map<String, Tensor>` (via `InferenceSession`)
3. **Postprocess**: `InferenceContext<I> → O` (via `Postprocessor`)

`InferenceContext<I>` carries data across stages — the original input, preprocessed tensors, and output tensors.

Exceptions: `SileroVadDetector` (stateful hidden state), `MiniLMSearchReranker`, and `ClipTextEncoder` do not extend `AbstractInferenceTask`.

## Result types

| Type | Fields | Used by |
|------|--------|---------|
| `Classification` | `label()`, `index()`, `confidence()` | `ResNetClassifier`, `EfficientNetClassifier` |
| `TextClassification` | `label()`, `classIndex()`, `confidence()` | `DistilBertTextClassifier` |
| `Detection` | `label()`, `classIndex()`, `confidence()`, `box()` | `YoloV8Detector`, `Yolo26Detector` |
| `TextRegion` | `box()`, `confidence()` | `CraftTextDetector` |
| `BoundingBox` | `x1()`, `y1()`, `x2()`, `y2()` | Embedded in `Detection`, `TextRegion` |
| `Transcription` | `text()` | `Wav2Vec2Recognizer` |
| `VoiceSegment` | `start()`, `end()`, `duration()`, `confidence()` | `SileroVadDetector` |

## Builder pattern

All task wrappers follow a consistent builder pattern:

```java
try (var task = SomeWrapper.builder()
        .modelId("org/model-name")              // HuggingFace model ID
        .modelSource(new LocalModelSource(...)) // optional: custom model source
        .sessionOptions(opts -> opts.addCoreML()) // optional: ONNX Runtime config
        // ... task-specific options ...
        .build()) {
    task.run(input);
}
```

The `.session(InferenceSession)` method exists on all builders but is package-private — the public API uses `modelId` + `modelSource` + `sessionOptions`.

## Functional interfaces

| Interface | Signature | Description |
|-----------|-----------|-------------|
| `SessionConfigurer` | `void configure(SessionOptions) throws OrtException` | ONNX Runtime session customization |
| `ModelSource` | `Path resolve(String modelId)` | Model file resolution |
| `Preprocessor<I, O>` | `O process(I input)` | Input transformation |
| `Postprocessor<I, O>` | `O process(I input)` | Output transformation |
| `OutputOperator` | `float[] apply(float[] values)` | Activation function (softmax, sigmoid, identity) |
