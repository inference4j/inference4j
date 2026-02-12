# inference4j Examples

Runnable examples demonstrating inference4j capabilities.

## Setup

### 1. Download the model

Text examples use [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (~90 MB). The router example also uses [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) (~120 MB), and the comparison example uses [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) (~420 MB).

The image classification example uses [ResNet-50](https://huggingface.co/onnxmodelzoo/resnet50-v1-7) ONNX (~98 MB) and [EfficientNet-Lite4](https://huggingface.co/onnx/EfficientNet-Lite4) ONNX (~49 MB), plus a sample image.

```bash
# From the project root:

# all-MiniLM-L6-v2 (required by all examples)
mkdir -p inference4j-examples/models/all-MiniLM-L6-v2
curl -L -o inference4j-examples/models/all-MiniLM-L6-v2/model.onnx \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
curl -L -o inference4j-examples/models/all-MiniLM-L6-v2/vocab.txt \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt

# all-MiniLM-L12-v2 (required by ModelRouterExample)
mkdir -p inference4j-examples/models/all-MiniLM-L12-v2
curl -L -o inference4j-examples/models/all-MiniLM-L12-v2/model.onnx \
  https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/onnx/model.onnx
curl -L -o inference4j-examples/models/all-MiniLM-L12-v2/vocab.txt \
  https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/vocab.txt

# all-mpnet-base-v2 (required by ModelComparisonExample)
mkdir -p inference4j-examples/models/all-mpnet-base-v2
curl -L -o inference4j-examples/models/all-mpnet-base-v2/model.onnx \
  https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/onnx/model.onnx
curl -L -o inference4j-examples/models/all-mpnet-base-v2/vocab.txt \
  https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/vocab.txt

# ResNet-50 (required by ImageClassificationExample)
mkdir -p inference4j-examples/models/resnet50
curl -L -o inference4j-examples/models/resnet50/model.onnx \
  "https://huggingface.co/onnxmodelzoo/resnet50-v1-7/resolve/main/resnet50-v1-7.onnx?download=true"

# EfficientNet-Lite4 (required by ImageClassificationExample)
mkdir -p inference4j-examples/models/efficientnet-lite4
curl -L -o inference4j-examples/models/efficientnet-lite4/model.onnx \
  "https://huggingface.co/onnx/EfficientNet-Lite4/resolve/main/efficientnet-lite4-11.onnx?download=true"

# Sample image for classification
mkdir -p inference4j-examples/images
curl -L -o inference4j-examples/images/sample.jpg \
  https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg
```

### 2. Run an example

```bash
# Semantic similarity — compare sentence pairs
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SemanticSimilarityExample

# Semantic search — query a document corpus
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SemanticSearchExample

# Model router — A/B test with metrics tracking
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ModelRouterExample

# Model comparison — same queries, two models side by side
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ModelComparisonExample

# Image classification — classify an image with ResNet-50
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ImageClassificationExample
```

## Examples

| Example | Description |
|---------|-------------|
| `SemanticSimilarityExample` | Encodes sentence pairs and computes cosine similarity scores |
| `SemanticSearchExample` | Encodes a document corpus, then ranks documents by relevance to queries |
| `ModelRouterExample` | A/B tests MiniLM-L6 vs L12 with round-robin routing and per-request metrics |
| `ModelComparisonExample` | Runs semantic search with two models and compares their rankings side by side |
| `ImageClassificationExample` | Classifies an image with ResNet-50 and EfficientNet-B0, printing top-5 predictions |
