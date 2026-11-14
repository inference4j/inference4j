# Embedding model

`Inference4jEmbeddingModel` is a LangChain4j `EmbeddingModel` backed by an inference4j
[`SentenceTransformerEmbedder`](../../use-cases/semantic-search.md), running in-process. Use it anywhere LangChain4j
takes an `EmbeddingModel`: `EmbeddingStoreIngestor`, `EmbeddingStoreContentRetriever`, or directly.

## Quick example

```java
var embeddingModel = Inference4jEmbeddingModel.builder()
        .embedder(SentenceTransformerEmbedder.builder()
                .modelId("inference4j/all-MiniLM-L6-v2").normalize().build())
        .build();

Response<Embedding> embedding = embeddingModel.embed("Hello, world!");
int dimension = embeddingModel.dimension();     // 384
```

## Query and document prefixes

Retrieval models such as nomic-embed, E5 and BGE are trained with one prefix on queries and another on documents.
A missing or wrong prefix lowers retrieval quality without raising an error.

The adapter reads LangChain4j's `EmbeddingInputType` from each request and applies the matching prefix, so one
model instance serves both ingestion and retrieval:

```java
var embeddingModel = Inference4jEmbeddingModel.builder()
        .embedder(SentenceTransformerEmbedder.builder()
                .modelId("inference4j/nomic-embed-text-v1.5").normalize().build())
        .queryPrefix("search_query: ")
        .documentPrefix("search_document: ")
        .build();

EmbeddingStoreIngestor.builder()
        .embeddingModel(embeddingModel)
        .embeddingStore(store)
        .embeddingInputType(EmbeddingInputType.DOCUMENT)      // "search_document: " + segment
        .build()
        .ingest(documents);

ContentRetriever retriever = EmbeddingStoreContentRetriever.builder()
        .embeddingModel(embeddingModel)
        .embeddingStore(store)
        .embeddingInputType(EmbeddingInputType.QUERY)         // "search_query: " + question
        .build();
```

!!! warning "Set `embeddingInputType` on both the ingestor and the retriever"
    LangChain4j sends no input type unless one is set. A request without one is embedded as the adapter's
    `defaultInputType` (`DOCUMENT` unless changed), so a retriever without `embeddingInputType(QUERY)` embeds
    questions with the document prefix. The first such request logs a warning when a prefix is configured.

Either prefix can be left unset. BGE, for example, only puts an instruction on queries:

```java
var embeddingModel = Inference4jEmbeddingModel.builder()
        .embedder(SentenceTransformerEmbedder.builder()
                .modelId("inference4j/bge-base-en-v1.5")
                .poolingStrategy(PoolingStrategy.CLS).normalize().build())
        .queryPrefix("Represent this sentence for searching relevant passages: ")
        .build();
```

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.embedder(SentenceTransformerEmbedder)` | `SentenceTransformerEmbedder` | — (required) | The model to run. Build it **without** `.textPrefix(...)`; the adapter's prefixes replace it on every call |
| `.queryPrefix(String)` | `String` | none | Prefix for `QUERY` inputs, e.g. `"search_query: "` (nomic) or `"query: "` (E5) |
| `.documentPrefix(String)` | `String` | none | Prefix for `DOCUMENT` inputs, e.g. `"search_document: "` (nomic) or `"passage: "` (E5) |
| `.defaultInputType(EmbeddingInputType)` | `EmbeddingInputType` | `DOCUMENT` | Input type for requests that carry none |
| `.dimension(int)` | `int` | probed | Size reported by `dimension()`. Without it, the adapter embeds one short text the first time `dimension()` is called. Must match the embedder's output, including its Matryoshka `.dimensions(int)` |
| `.modelName(String)` | `String` | LangChain4j's default | Name reported by `modelName()` and in response metadata |

Model settings (pooling, normalization, maximum length, long-input windows, Matryoshka size, hardware
acceleration) are set on the embedder. See its
[builder options](../../use-cases/semantic-search.md#embedder-builder-options).

## Model choices

| Model ID | Dimensions | Size | Prefixes | Notes |
|----------|-----------|------|----------|-------|
| `inference4j/all-MiniLM-L6-v2` | 384 | ~90 MB | none | Also packaged by LangChain4j as `AllMiniLmL6V2EmbeddingModel` |
| `inference4j/bge-base-en-v1.5` | 768 | ~430 MB | query only | Use `PoolingStrategy.CLS` |
| `inference4j/gte-base` | 768 | ~430 MB | none | |
| `inference4j/nomic-embed-text-v1.5` | 768, or 512 / 256 / 128 / 64 | ~520 MB | `search_query: ` / `search_document: ` | Reads up to 8,192 tokens with `.maxLength(8192)`; Matryoshka sizes with `.dimensions(int)` |

More models and their settings are listed under
[Semantic search](../../use-cases/semantic-search.md) and [Supported models](../../reference/supported-models.md).

### Smaller vectors (Matryoshka)

nomic-embed-text-v1.5 vectors can be shortened with little loss, which reduces index size and search time. Set the
size on the embedder and on the adapter:

```java
var embeddingModel = Inference4jEmbeddingModel.builder()
        .embedder(SentenceTransformerEmbedder.builder()
                .modelId("inference4j/nomic-embed-text-v1.5").normalize().dimensions(256).build())
        .queryPrefix("search_query: ")
        .documentPrefix("search_document: ")
        .dimension(256)
        .build();
```

## Notes

- **Normalize.** Build the embedder with `.normalize()` when the embedding store compares by cosine similarity.
- **Per-call `dimensions` and `modelName` are rejected.** The only request parameter the adapter supports is
  `inputType`; LangChain4j fails a request that sets any other with `UnsupportedFeatureException`.
- **Batches run sequentially.** `embedAll` embeds one input at a time; each run uses all cores. See
  [Choosing an embedding or scoring model](index.md#choosing-an-embedding-or-scoring-model).
- **Long segments.** Text over the embedder's `maxLength` (512 tokens by default) is truncated with a warning. Split
  documents into segments that fit, or set `.stride(int)` on the embedder to average windows; with `stride`, the
  prefix is repeated in every window.
- **Token usage is not reported.** `EmbeddingResponse` carries embeddings and the model name only.
- **Lifecycle.** `close()` closes the embedder. See
  [Lifecycle](index.md#lifecycle).
