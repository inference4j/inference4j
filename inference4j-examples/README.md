# inference4j Examples

Runnable examples demonstrating inference4j capabilities.

## Setup

### 1. Download the model

These examples use [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a lightweight sentence embedding model (~90 MB).

```bash
# From the project root:
mkdir -p inference4j-examples/models/all-MiniLM-L6-v2

# Download model and vocabulary
curl -L -o inference4j-examples/models/all-MiniLM-L6-v2/model.onnx \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx

curl -L -o inference4j-examples/models/all-MiniLM-L6-v2/vocab.txt \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt
```

### 2. Run an example

```bash
# Semantic similarity — compare sentence pairs
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SemanticSimilarityExample

# Semantic search — query a document corpus
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SemanticSearchExample
```

## Examples

| Example | Description |
|---------|-------------|
| `SemanticSimilarityExample` | Encodes sentence pairs and computes cosine similarity scores |
| `SemanticSearchExample` | Encodes a document corpus, then ranks documents by relevance to queries |
