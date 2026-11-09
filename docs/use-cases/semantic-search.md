# Semantic Search

Build a semantic search pipeline using text embeddings and cross-encoder reranking. inference4j provides two complementary models: `SentenceTransformerEmbedder` for fast retrieval and `MiniLMSearchReranker` for precision reranking.

## Quick example — embeddings

```java
try (var embedder = SentenceTransformerEmbedder.builder()
        .modelId("inference4j/all-MiniLM-L6-v2").build()) {
    float[] embedding = embedder.encode("Hello, world!");
}
```

## Quick example — reranking

```java
try (var reranker = MiniLMSearchReranker.builder().build()) {
    float score = reranker.score("What is Java?", "Java is a programming language.");
}
```

## Full search pipeline

A typical semantic search pipeline uses embeddings for fast candidate retrieval, then a cross-encoder reranker for precision scoring of the top results.

```java
import io.github.inference4j.nlp.SentenceTransformerEmbedder;
import io.github.inference4j.nlp.MiniLMSearchReranker;

public class SemanticSearch {
    public static void main(String[] args) {
        String query = "How do I handle errors in Java?";
        List<String> documents = List.of(
            "Java uses try-catch blocks for exception handling.",
            "Python decorators are a powerful feature.",
            "Error handling in Java includes checked and unchecked exceptions.",
            "The Java Stream API provides functional operations on collections."
        );

        // Stage 1: Embed and retrieve candidates by cosine similarity
        try (var embedder = SentenceTransformerEmbedder.builder()
                .modelId("inference4j/all-MiniLM-L6-v2").build()) {

            float[] queryEmbedding = embedder.encode(query);
            List<float[]> docEmbeddings = embedder.encodeBatch(documents);

            // Rank by cosine similarity (your own similarity function)
            // ...
        }

        // Stage 2: Rerank top candidates with cross-encoder
        try (var reranker = MiniLMSearchReranker.builder().build()) {
            float[] scores = reranker.scoreBatch(query, documents);

            for (int i = 0; i < documents.size(); i++) {
                System.out.printf("%.4f  %s%n", scores[i], documents.get(i));
            }
        }
    }
}
```

## Embedder builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | — (required) | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.tokenizer(Tokenizer)` | `Tokenizer` | auto-loaded `WordPieceTokenizer` | Custom tokenizer |
| `.poolingStrategy(PoolingStrategy)` | `PoolingStrategy` | `MEAN` | Pooling method: `CLS`, `MEAN`, or `MAX` |
| `.normalize()` | — | disabled | Enables L2 normalization of output embeddings |
| `.textPrefix(String)` | `String` | `null` | Text prefix to prepend before encoding |
| `.maxLength(int)` | `int` | `512` | Maximum token sequence length |
| `.truncation(TruncationPolicy)` | `TruncationPolicy` | `TRUNCATE` | Input longer than the token limit: `TRUNCATE` keeps the first tokens and logs a warning, `FAIL` throws `InputTooLongException`. See [Input length](../reference/configuration.md#input-length) |
| `.stride(int)` | `int` | off | Embed input longer than `maxLength` in full: windows sharing this many tokens (`0` = consecutive chunks), averaged by the tokens each covers |
| `.dimensions(int)` | `int` | off (full size) | Matryoshka truncation: keep the first N values, re-normalized when `.normalize()` is on. Only for models trained for it (nomic-embed-text-v1.5) |

## Reranker builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/ms-marco-MiniLM-L-6-v2` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.tokenizer(Tokenizer)` | `Tokenizer` | auto-loaded `WordPieceTokenizer` | Custom tokenizer |
| `.maxLength(int)` | `int` | `512` | Maximum token sequence length |
| `.truncation(TruncationPolicy)` | `TruncationPolicy` | `TRUNCATE` | Input longer than the token limit: `TRUNCATE` keeps the first tokens and logs a warning, `FAIL` throws `InputTooLongException`. See [Input length](../reference/configuration.md#input-length) |
| `.stride(int)` | `int` | off | Score documents longer than `maxLength` with MaxP: passages sharing this many tokens, each scored with the full query; the highest score wins |

## Input length

| Task | Limit | When exceeded | Long-input strategies |
|------|-------|---------------|-----------------------|
| Embedder | `maxLength` tokens (512 by default) | Truncated with a warning (default), or rejected with `.truncation(TruncationPolicy.FAIL)` | `.stride(int)`: windows averaged into one vector |
| Reranker | `maxLength` tokens (512 by default) for query and document combined; the longer one is shortened first | Truncated with a warning (default), or rejected with `.truncation(TruncationPolicy.FAIL)` | `.stride(int)`: MaxP over passages |

An embedding of a truncated text only represents its beginning. There are two ways to cover long documents:

- **Split documents into chunks before embedding** (for example with a document splitter), and store one vector per chunk. This retrieves best: every passage keeps its own vector.
- **Use `.stride(int)`** when you need exactly one vector per document. The text is split into windows, each window is embedded, and the vectors are averaged, weighted by the tokens each window covers. This matches what LangChain4j's in-process embedding models do.

```java
try (var embedder = SentenceTransformerEmbedder.builder()
        .modelId("inference4j/all-MiniLM-L6-v2")
        .normalize()
        .stride(0)          // consecutive chunks; use e.g. 64 for overlapping windows
        .build()) {
    float[] documentVector = embedder.encode(longDocument);
}
```

An averaged vector represents each part of the document in proportion to its length. A topic covered by one sentence in a long document carries little weight, which is why chunking is usually better for search. In a test with about 550 tokens about baking followed by about 250 about whales, a query about whales scored 0.04 cosine similarity against the truncated embedding and 0.21 against the averaged one.

### Reranking long documents (MaxP)

With `.stride(int)`, a document too long for one pass is split into passages, each passage is scored together with the full query, and the highest passage score is the document's score. This is the established "MaxP" technique.

```java
try (var reranker = MiniLMSearchReranker.builder()
        .maxLength(128)
        .stride(32)
        .build()) {
    float score = reranker.score("How many people live in Berlin?", longDocument);
}
```

!!! warning "Keep passages short"
    `ms-marco-MiniLM-L-6-v2` was trained on short MS MARCO passages, and a relevant sentence carries less weight the more unrelated text surrounds it. To measure this, we appended the sentence *"Berlin has a population of about 3.7 million people…"* to the end of 10 real news articles (each over 520 tokens), and scored them for the query *"How many people live in Berlin?"*:

    | Setting | Mean score |
    |---|---|
    | No windowing (truncates at 512) | 0.000 |
    | `stride(128)` with the default `maxLength(512)` | 0.076 |
    | `maxLength(256).stride(64)` | 0.001 |
    | **`maxLength(128).stride(32)`** | **0.962** (lowest 0.872) |
    | The same articles without the sentence, MaxP | 0.000 |

    Use `maxLength(128).stride(32)` for MaxP. The effect also applies without MaxP: when reranking retrieved chunks, shorter chunks (around 100–150 tokens) let a relevant passage stand out. With varied text, the drop is gradual: the same sentence preceded by about 150 to 240 tokens of other text still scored 0.80–0.92 on its own.

## Result types

### Embeddings

`encode()` returns a `float[]` — a dense vector representation of the input text. Use cosine similarity to compare embeddings:

```java
static double cosineSimilarity(float[] a, float[] b) {
    double dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

### Reranking scores

`score()` returns a `float` in the range [0, 1] — the sigmoid-normalized relevance score for a query-document pair. Higher means more relevant.

`scoreBatch()` returns a `float[]` — one score per document, scored against the same query.

## Pooling strategies

The embedder supports three pooling strategies for converting token-level representations into a single sentence embedding:

| Strategy | Description |
|----------|-------------|
| `MEAN` | Average of all token embeddings (default, best for most tasks) |
| `CLS` | Uses only the `[CLS]` token embedding |
| `MAX` | Element-wise maximum across all token embeddings |

## Available embedding models

| Model | Dim | Pooling | Normalize | Text Prefix | MTEB Avg |
|-------|-----|---------|-----------|-------------|----------|
| `inference4j/all-MiniLM-L6-v2` | 384 | MEAN | Optional | — | 56.26 |
| `inference4j/all-mpnet-base-v2` | 768 | MEAN | Optional | — | 57.78 |
| `inference4j/bge-base-en-v1.5` | 768 | CLS | Recommended | — | 63.55 |
| `inference4j/gte-base` | 768 | CLS | Recommended | — | 61.36 |
| `inference4j/nomic-embed-text-v1.5` | 768 (Matryoshka: 512, 256, 128, 64) | MEAN | Recommended | `search_query: ` / `search_document: ` | 62.28 |

### BGE example

```java
try (var embedder = SentenceTransformerEmbedder.builder()
        .modelId("inference4j/bge-base-en-v1.5")
        .poolingStrategy(PoolingStrategy.CLS)
        .normalize()
        .build()) {
    float[] embedding = embedder.encode("What is machine learning?");
}
```

### Nomic example (long context, prefixes, smaller vectors)

nomic-embed-text-v1.5 reads up to 8,192 tokens, so raise `maxLength` for long documents. It needs a task prefix: `"search_query: "` for queries and `"search_document: "` for documents, so use one embedder per side.

```java
try (var queries = SentenceTransformerEmbedder.builder()
             .modelId("inference4j/nomic-embed-text-v1.5")
             .textPrefix("search_query: ")
             .normalize()
             .build();
     var documents = SentenceTransformerEmbedder.builder()
             .modelId("inference4j/nomic-embed-text-v1.5")
             .textPrefix("search_document: ")
             .maxLength(8192)
             .normalize()
             .build()) {
    float[] query = queries.encode("What is the capital of France?");
    float[] document = documents.encode("Paris is the capital and largest city of France.");
}
```

The model was trained with Matryoshka representation learning, so `.dimensions(int)` can shorten its vectors with little loss. Smaller vectors cut index size and search time. Use the same `dimensions` for queries and documents.

| Dimensions | MTEB Avg |
|------------|----------|
| 768 | 62.28 |
| 512 | 61.96 |
| 256 | 61.04 |
| 128 | 59.34 |
| 64 | 56.10 |

Scores are from the [model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5). The card applies a layer norm before truncating. inference4j truncates and re-normalizes without it; the two match to a cosine similarity of 0.9999 or better on our reference set.

### E5 example (with text prefix)

E5 models require a text prefix: `"query: "` for queries, `"passage: "` for documents.

```java
try (var queryEncoder = SentenceTransformerEmbedder.builder()
        .modelId("inference4j/e5-base-v2")
        .textPrefix("query: ")
        .normalize()
        .build()) {
    float[] queryEmbedding = queryEncoder.encode("What is Java?");
}
```

## Tips

- **Two-stage pipeline**: Use embeddings for fast top-K retrieval (cheap cosine similarity), then rerank the top candidates with the cross-encoder (expensive but more accurate).
- **Batch encoding**: Use `encodeBatch()` when encoding multiple texts — more efficient than calling `encode()` in a loop.
- **L2 normalization**: Enable `.normalize()` when comparing embeddings with cosine similarity. BGE, GTE, and E5 models all recommend normalization.
- **Text prefix**: Some models (E5, Nomic) require a text prefix. Check the model card for the correct prefix.
- Embedding dimension depends on the model: all-MiniLM-L6-v2 produces 384-dimensional vectors, while BGE/GTE/mpnet/Nomic produce 768-dimensional vectors. Nomic can be shortened with `.dimensions(int)`.
