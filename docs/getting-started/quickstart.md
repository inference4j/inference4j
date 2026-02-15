# Quick Start

## Your first model in 3 lines

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

## What happens under the hood

When you call `builder().build()`:

1. **Model resolution** — The builder resolves the default model ID (`inference4j/distilbert-base-uncased-finetuned-sst-2-english`) using `HuggingFaceModelSource`, which downloads the ONNX model, vocabulary, and config files to `~/.cache/inference4j/`
2. **Session creation** — An ONNX Runtime `InferenceSession` is created with default session options
3. **Tokenizer setup** — A `WordPieceTokenizer` is loaded from the downloaded `vocab.txt`

When you call `classify(text)`:

1. **Preprocessing** — The text is tokenized into input IDs and attention mask tensors
2. **Inference** — Tensors are passed to the ONNX Runtime session for a single forward pass
3. **Postprocessing** — Raw logits are transformed via softmax into labeled classifications

## Try another domain

### Vision

```java
import io.github.inference4j.vision.ResNetClassifier;

try (var classifier = ResNetClassifier.builder().build()) {
    var results = classifier.classify(Path.of("cat.jpg"));
    results.forEach(c ->
        System.out.printf("%s: %.2f%%%n", c.label(), c.confidence() * 100));
}
```

### Audio

```java
import io.github.inference4j.audio.Wav2Vec2Recognizer;

try (var recognizer = Wav2Vec2Recognizer.builder().build()) {
    System.out.println(recognizer.transcribe(Path.of("speech.wav")).text());
}
```

## Customizing the builder

Every model wrapper follows the same builder pattern:

```java
try (var classifier = ResNetClassifier.builder()
        .modelId("inference4j/resnet50-v1-7")     // override default model
        .sessionOptions(opts -> opts.addCoreML())  // hardware acceleration
        .defaultTopK(3)                            // return top 3 predictions
        .build()) {
    classifier.classify(Path.of("cat.jpg"));
}
```

Common builder methods available on all wrappers:

| Method | Description |
|--------|-------------|
| `.modelId(String)` | Override the default HuggingFace model ID |
| `.modelSource(ModelSource)` | Use a custom model source (e.g., `LocalModelSource`) |
| `.sessionOptions(SessionConfigurer)` | Configure ONNX Runtime options (GPU, threads) |

## Next steps

- Browse the [Use Cases](../use-cases/sentiment-analysis.md) for detailed examples of each capability
- See [Hardware Acceleration](../guides/hardware-acceleration.md) for GPU and CoreML setup
- Check [Supported Models](../reference/supported-models.md) for all available models
