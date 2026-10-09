# Configuration

## Model cache

Models are downloaded from HuggingFace and cached locally. The cache directory is resolved in this order:

| Priority | Method | Example |
|----------|--------|---------|
| 1 | Constructor parameter | `new HuggingFaceModelSource(Path.of("/cache"))` |
| 2 | System property | `-Dinference4j.cache.dir=/path/to/cache` |
| 3 | Environment variable | `INFERENCE4J_CACHE_DIR=/path/to/cache` |
| 4 | Default | `~/.cache/inference4j/` |

## JVM flags

ONNX Runtime requires native access:

```
--enable-native-access=ALL-UNNAMED
```

Or, on the module path:

```
--enable-native-access=com.microsoft.onnxruntime
```

## System properties

| Property | Description | Default |
|----------|-------------|---------|
| `inference4j.cache.dir` | Model cache directory | `~/.cache/inference4j/` |

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE4J_CACHE_DIR` | Model cache directory | `~/.cache/inference4j/` |

## Spring Boot properties

See the [Spring Boot guide](../guides/spring-boot.md#all-properties) for the full list of `inference4j.*` application properties.

## Input length

Models have a fixed maximum input length in tokens: 512 for the BERT-family NLP models, 77 for the CLIP text encoder, 256 for `PunctCapSegModel`. Text generators read their limit from the model's `config.json` (1024 for BART, 512 for MarianMT and T5; for decoder-only models, the model's positions minus `maxNewTokens`) and accept `.maxInputLength(int)` to change it. Each affected use-case page lists its limit. The builders of these tasks accept a `TruncationPolicy` that controls what happens to longer input:

| Policy | Behavior |
|--------|----------|
| `TRUNCATE` (default) | Keeps the first tokens and drops the rest. The first truncation per task instance is logged at `WARN` with the token counts; later ones at `DEBUG`. |
| `FAIL` | Throws `InputTooLongException`, which reports `tokenCount()` and `maxTokens()`. |

```java
try (var ner = BertNerRecognizer.builder()
        .truncation(TruncationPolicy.FAIL)
        .build()) {
    ner.recognize(longDocument);   // throws InputTooLongException instead of silently ignoring the end
}
```

`FAIL` is useful in tests and in pipelines where losing part of the input is unacceptable, such as PII redaction. Tasks that can process long input in full expose that as a separate builder option, listed under *Long-input strategies* on their page.

## ONNX Runtime session options

Session-level configuration is set via `.sessionOptions()` on each builder:

```java
.sessionOptions(opts -> {
    opts.addCoreML();                                      // execution provider
    opts.setIntraOpNumThreads(4);                          // parallelism
    opts.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT); // graph optimization
})
```

See the [Hardware Acceleration guide](../guides/hardware-acceleration.md) for execution provider details.
