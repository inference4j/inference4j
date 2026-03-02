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

## Reranker builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/ms-marco-MiniLM-L-6-v2` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.tokenizer(Tokenizer)` | `Tokenizer` | auto-loaded `WordPieceTokenizer` | Custom tokenizer |
| `.maxLength(int)` | `int` | `512` | Maximum token sequence length |

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
- Embedding dimension depends on the model: all-MiniLM-L6-v2 produces 384-dimensional vectors, while BGE/GTE/mpnet produce 768-dimensional vectors.
