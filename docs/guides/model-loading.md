# Model Loading

inference4j resolves models through the `ModelSource` interface. By default, models are downloaded from HuggingFace and cached locally. You can also load models from disk or implement your own resolution strategy.

## ModelSource interface

```java
@FunctionalInterface
public interface ModelSource {
    Path resolve(String modelId);
}
```

`ModelSource` takes a model ID and returns the local directory path where the model files live. Every builder accepts `.modelSource(ModelSource)`.

## LocalModelSource

Load models from a directory on disk:

```java
var source = new LocalModelSource(Path.of("/models"));

try (var classifier = DistilBertTextClassifier.builder()
        .modelId("my-sentiment-model")
        .modelSource(source)
        .build()) {
    classifier.classify("Great product!");
}
```

This resolves to `/models/my-sentiment-model/`, which should contain the model files (`model.onnx`, `vocab.txt`, `config.json`, etc.).

## HuggingFaceModelSource

The default model source. Downloads models from the [HuggingFace Hub](https://huggingface.co/) and caches them locally.

```java
// Uses default cache directory (~/.cache/inference4j/)
var source = HuggingFaceModelSource.defaultInstance();

// Or specify a custom cache directory
var source = new HuggingFaceModelSource(Path.of("/custom/cache"));
```

### Cache directory resolution

The cache directory is resolved in this order:

1. Constructor parameter (`new HuggingFaceModelSource(path)`)
2. System property: `-Dinference4j.cache.dir=/path/to/cache`
3. Environment variable: `INFERENCE4J_CACHE_DIR=/path/to/cache`
4. Default: `~/.cache/inference4j/`

### Downloaded files

`HuggingFaceModelSource` attempts to download these files (skipping any that don't exist):

- `model.onnx` — the ONNX model
- `vocab.txt` — WordPiece vocabulary (NLP models)
- `vocab.json` — JSON vocabulary (audio models)
- `config.json` — HuggingFace model config
- `labels.txt` — class labels (vision models)
- `silero_vad.onnx` — alternate model filename (Silero VAD)

## Lambda shorthand

Since `ModelSource` is a `@FunctionalInterface`, you can use a lambda:

```java
try (var classifier = ResNetClassifier.builder()
        .modelId("resnet50")
        .modelSource(id -> Path.of("/models").resolve(id))
        .build()) {
    classifier.classify(Path.of("cat.jpg"));
}
```

## Required model files by task

| Task | Required files |
|------|----------------|
| Text classification | `model.onnx`, `vocab.txt`, `config.json` (with `id2label`) |
| Text embeddings | `model.onnx`, `vocab.txt` |
| Search reranking | `model.onnx`, `vocab.txt` |
| Image classification | `model.onnx`, `labels.txt` |
| Object detection | `model.onnx`, `labels.txt` |
| Text detection | `model.onnx` |
| Speech-to-text | `model.onnx`, `vocab.json` |
| Voice activity detection | `silero_vad.onnx` |

## Using a custom HuggingFace model

Any ONNX-exported model hosted on HuggingFace works — just override the model ID:

```java
try (var classifier = DistilBertTextClassifier.builder()
        .modelId("your-org/your-fine-tuned-model")
        .build()) {
    classifier.classify("Some text");
}
```

The model repository must contain the required files listed above.

## Tips

- `LocalModelSource` does no downloading — it expects files to already exist at the resolved path.
- `HuggingFaceModelSource` is thread-safe. Concurrent requests for the same model wait for the first download to complete.
- Models are cached permanently. To re-download, delete the cached directory.
- The model ID is just a string — it can be any identifier meaningful to your `ModelSource` implementation.
