# inference4j Examples

Runnable examples demonstrating inference4j capabilities.

## Setup

### 1. Download the model

Text examples use [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (~90 MB). The router example also uses [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) (~120 MB), and the comparison example uses [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) (~420 MB).

The image classification example uses [ResNet-50](https://huggingface.co/onnxmodelzoo/resnet50-v1-7) ONNX (~98 MB) and [EfficientNet-Lite4](https://huggingface.co/onnx/EfficientNet-Lite4) ONNX (~49 MB), plus a sample image.

The object detection example uses [YOLOv8n](https://huggingface.co/Kalray/yolov8) ONNX (~13 MB) and [YOLO26n](https://huggingface.co/onnx-community/yolo26n-ONNX) ONNX (~18 MB), and reuses the sample image above.

The speech-to-text example uses [wav2vec2-base-960h](https://huggingface.co/Xenova/wav2vec2-base-960h) ONNX (~360 MB) and a sample WAV file (16kHz mono).

The voice activity detection example uses [Silero VAD](https://github.com/snakers4/silero-vad) ONNX (~2 MB), and reuses the sample audio from above.

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

# YOLOv8n (required by ObjectDetectionExample)
mkdir -p inference4j-examples/models/yolov8n
curl -L -o inference4j-examples/models/yolov8n/model.onnx \
  "https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx?download=true"

# YOLO26n (required by ObjectDetectionExample)
mkdir -p inference4j-examples/models/yolo26n
curl -L -o inference4j-examples/models/yolo26n/model.onnx \
  "https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/main/onnx/model.onnx?download=true"

# wav2vec2-base-960h (required by SpeechToTextExample)
mkdir -p inference4j-examples/models/wav2vec2-base-960h
curl -L -o inference4j-examples/models/wav2vec2-base-960h/model.onnx \
  "https://huggingface.co/Xenova/wav2vec2-base-960h/resolve/main/onnx/model.onnx?download=true"
curl -L -o inference4j-examples/models/wav2vec2-base-960h/vocab.json \
  "https://huggingface.co/Xenova/wav2vec2-base-960h/resolve/main/vocab.json?download=true"

# Sample audio for speech-to-text (LibriSpeech sample, 16kHz mono WAV)
mkdir -p inference4j-examples/audio
curl -L -o inference4j-examples/audio/sample.wav \
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav?download=true"

# Silero VAD (required by VoiceActivityDetectionExample)
mkdir -p inference4j-examples/models/silero-vad
curl -L -o inference4j-examples/models/silero-vad/model.onnx \
  "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
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

# Object detection — detect objects with YOLOv8n
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ObjectDetectionExample

# Speech-to-text — transcribe audio with Wav2Vec2
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.SpeechToTextExample

# Voice activity detection — detect speech segments with Silero VAD
./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.VoiceActivityDetectionExample
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
| `SpeechToTextExample` | Transcribes a WAV audio file to text using Wav2Vec2-CTC |
| `VoiceActivityDetectionExample` | Detects speech segments in audio using Silero VAD |
