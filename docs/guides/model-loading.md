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

// Or set every option
var source = HuggingFaceModelSource.builder()
        .cacheDir(Path.of("/custom/cache"))
        .checkInterval(Duration.ofHours(1))   // default: a day
        .offline(false)
        .endpoint(URI.create("https://hf-mirror.example.com"))
        .build();
```

### Updates

Cached models stay in sync with the Hub. At most once per check interval (a day by default), resolving a
cached model asks the Hub for the repo's current revision, with one small API call. Files whose content
changed on the Hub are downloaded again, at that revision; unchanged files are kept, so a model card edit
never re-downloads a model. The commit and each file's id are recorded in `.inference4j-revision.json`
in the model's cache directory.

- **Caches from before 0.15.0** have no record yet. On the first check, the cached files are hashed once
  and compared with the Hub's, and only those that differ are downloaded again.
- **Hub unreachable:** if the check fails (no network, timeout, an error response), the cached files are
  used and a warning is logged once. Missing files still need the Hub.
- **Offline mode** never contacts the Hub: cached files are used as they are, and a model that is not fully
  cached fails to load. Turn it on with `INFERENCE4J_OFFLINE=true`, `-Dinference4j.offline=true`,
  `HF_HUB_OFFLINE=1` or `.offline(true)`.
- **Check interval:** `INFERENCE4J_CACHE_CHECK_INTERVAL` or `-Dinference4j.cache.check-interval` take an
  ISO-8601 duration, such as `PT1H` or `P7D`; `PT0S` checks on every load.
- **Mirrors:** `HF_ENDPOINT` or `-Dinference4j.hf.endpoint` point to a Hub mirror.

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
- Cached models are checked for updates once a day (see [Updates](#updates)). To force a full re-download,
  delete the model's cached directory.
- The model ID is just a string — it can be any identifier meaningful to your `ModelSource` implementation.
