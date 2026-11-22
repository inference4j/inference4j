# Configuration

## Model cache

Models are downloaded from HuggingFace and cached locally. The cache directory is resolved in this order:

| Priority | Method | Example |
|----------|--------|---------|
| 1 | Constructor parameter | `new HuggingFaceModelSource(Path.of("/cache"))` |
| 2 | System property | `-Dinference4j.cache.dir=/path/to/cache` |
| 3 | Environment variable | `INFERENCE4J_CACHE_DIR=/path/to/cache` |
| 4 | Default | `~/.cache/inference4j/` |

Each repo is cached as snapshots under `{cacheDir}/hub/`, one per Hub commit. Cached models are checked against
the Hub at most once a day, and files that changed there are downloaded into a new snapshot; see
[Model loading](../guides/model-loading.md#updates). Superseded snapshots are pruned after each update unless a
loaded model uses them; see [Pruning and snapshots in use](../guides/model-loading.md#pruning-and-snapshots-in-use).

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
| `inference4j.cache.check-interval` | How often cached models are checked for updates (ISO-8601 duration) | `P1D` |
| `inference4j.cache.prune` | `auto` deletes superseded snapshots after each update unless in use; `manual` keeps them until `ModelCache.prune()` | `auto` |
| `inference4j.offline` | Never contact the Hub; use cached files as they are | `false` |
| `inference4j.hf.endpoint` | Hugging Face Hub endpoint, such as a mirror | `https://huggingface.co` |

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE4J_CACHE_DIR` | Model cache directory | `~/.cache/inference4j/` |
| `INFERENCE4J_CACHE_CHECK_INTERVAL` | How often cached models are checked for updates (ISO-8601 duration) | `P1D` |
| `INFERENCE4J_CACHE_PRUNE` | `auto` or `manual`, like `inference4j.cache.prune` | `auto` |
| `INFERENCE4J_OFFLINE` | Never contact the Hub; use cached files as they are | `false` |
| `HF_HUB_OFFLINE` | Same as `INFERENCE4J_OFFLINE` when set to `1` (the Hugging Face convention) | unset |
| `HF_ENDPOINT` | Hugging Face Hub endpoint, such as a mirror | `https://huggingface.co` |

## Spring Boot properties

See the [Spring Boot guide](../guides/spring-boot.md#all-properties) for the full list of `inference4j.*` application properties.

## Input length

Models have a fixed maximum input length in tokens: 512 for the BERT-family NLP models, 77 for the CLIP text encoder, 256 for `PunctCapSegModel`. Text generators read their limit from the model's `config.json` (1024 for BART, 512 for MarianMT and T5; for decoder-only models, the model's positions minus `maxNewTokens`) and accept `.maxInputLength(int)` to change it. Encoder-decoder models with learned or sinusoidal positions (BART, MarianMT) also cap `maxNewTokens` at their decoder's positions, with a warning; T5's relative positions have no cap. Each affected use-case page lists its limit. The builders of these tasks accept a `TruncationPolicy` that controls what happens to longer input:

| Policy | Behavior |
|--------|----------|
| `TRUNCATE` (default) | Keeps the first tokens and drops the rest; decoder-only text generators instead keep the chat template and the end of the message (see [Prompt length](../generative-ai/native-text-generation.md#prompt-length)). The first truncation per task instance is logged at `WARN` with the token counts; later ones at `DEBUG`. |
| `FAIL` | Throws `InputTooLongException`, which reports `tokenCount()` and `maxTokens()`. |

```java
try (var ner = BertNerRecognizer.builder()
        .truncation(TruncationPolicy.FAIL)
        .build()) {
    ner.recognize(longDocument);   // throws InputTooLongException instead of silently ignoring the end
}
```

`FAIL` is useful in tests and in pipelines where losing part of the input is unacceptable, such as PII redaction. Tasks that can process long input in full expose that as a separate builder option, listed under *Long-input strategies* on their page. Currently, [named entity recognition](../use-cases/named-entity-recognition.md#long-documents), [punctuation & casing](../use-cases/punctuation-and-casing.md#input-length) the [embedder](../use-cases/semantic-search.md#input-length), the [reranker](../use-cases/semantic-search.md#reranking-long-documents-maxp) and the [text classifier](../use-cases/sentiment-analysis.md#long-texts) support `.stride(int)`, which processes the whole input in windows. The classifier combines its windows' scores with a pluggable `WindowAggregation`. For [translation](../use-cases/translation.md#long-documents) and [grammar correction](../use-cases/grammar-correction.md#long-documents), the `ChunkedTranslator` and `ChunkedGrammarCorrector` decorators cut long text into sentence-aligned chunks that fit the model, using the token budget the wrapped task reports as `TokenLimits`.

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
