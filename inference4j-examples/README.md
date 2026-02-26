# inference4j Examples

Runnable examples demonstrating inference4j capabilities.

## Setup

### Models

**Models are auto-downloaded.** When you run an example, inference4j automatically downloads the required model from [HuggingFace](https://huggingface.co/inference4j) and caches it in `~/.cache/inference4j/`. No manual setup required — just run the example.

Cache location can be customized via:
- System property: `-Dinference4j.cache.dir=/path/to/cache`
- Environment variable: `INFERENCE4J_CACHE_DIR=/path/to/cache`

### Run an example

Sample images and audio files are bundled in `src/main/resources/fixtures/` — no manual download needed.

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

# Object detection — detect objects with YOLOv8n
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ObjectDetectionExample

# Text classification — sentiment analysis with DistilBERT
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.TextClassificationExample

# Cross-encoder reranking — rerank search results with MiniLM
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.CrossEncoderRerankerExample

# Speech-to-text — transcribe audio with Wav2Vec2
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SpeechToTextExample

# Voice activity detection — detect speech segments with Silero VAD
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.VoiceActivityDetectionExample

# Text detection — detect text regions with CRAFT
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.CraftTextDetectionExample
```

## Examples

| Example | Description |
|---------|-------------|
| `SemanticSimilarityExample` | Encodes sentence pairs and computes cosine similarity scores |
| `SemanticSearchExample` | Encodes a document corpus, then ranks documents by relevance to queries |
| `ModelRouterExample` | A/B tests MiniLM-L6 vs L12 with round-robin routing and per-request metrics |
| `ModelComparisonExample` | Runs semantic search with two models and compares their rankings side by side |
| `ImageClassificationExample` | Classifies an image with ResNet-50 and EfficientNet-B0, printing top-5 predictions |
| `ObjectDetectionExample` | Detects objects in an image with YOLOv8n and YOLO26n, printing bounding boxes and labels |
| `TextClassificationExample` | Classifies text sentiment with DistilBERT fine-tuned on SST-2 |
| `CrossEncoderRerankerExample` | Reranks search result candidates using ms-marco-MiniLM cross-encoder |
| `SpeechToTextExample` | Transcribes a WAV audio file to text using Wav2Vec2-CTC |
| `VoiceActivityDetectionExample` | Detects speech segments in audio using Silero VAD |
| `CraftTextDetectionExample` | Detects text regions in an image using CRAFT |
