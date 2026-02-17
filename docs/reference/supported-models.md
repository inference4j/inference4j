# Supported Models

All supported models are hosted under the [`inference4j`](https://huggingface.co/inference4j) HuggingFace organization and are automatically downloaded and cached on first use.

## NLP

| Capability | Wrapper | Default Model ID | Size | API |
|------------|---------|-------------------|------|-----|
| Text Classification | `DistilBertTextClassifier` | `inference4j/distilbert-base-uncased-finetuned-sst-2-english` | ~260 MB | `TextClassifier` |
| Text Embeddings | `SentenceTransformerEmbedder` | `inference4j/all-MiniLM-L6-v2` | ~90 MB | `TextEmbedder` |
| Search Reranking | `MiniLMSearchReranker` | `inference4j/ms-marco-MiniLM-L-6-v2` | ~90 MB | `SearchReranker` |

## Vision

| Capability | Wrapper | Default Model ID | Size | API |
|------------|---------|-------------------|------|-----|
| Image Classification | `ResNetClassifier` | `inference4j/resnet50-v1-7` | ~100 MB | `ImageClassifier` |
| Image Classification | `EfficientNetClassifier` | `inference4j/efficientnet-lite4` | ~50 MB | `ImageClassifier` |
| Object Detection | `YoloV8Detector` | `inference4j/yolov8n` | ~25 MB | `ObjectDetector` |
| Object Detection | `Yolo26Detector` | `inference4j/yolo26n` | ~25 MB | `ObjectDetector` |
| Text Detection | `CraftTextDetector` | `inference4j/craft-mlt-25k` | ~80 MB | `TextDetector` |

## Multimodal

| Capability | Wrapper | Default Model ID | Size | API |
|------------|---------|-------------------|------|-----|
| Zero-Shot Classification | `ClipClassifier` | `inference4j/clip-vit-base-patch32` | ~595 MB | `ZeroShotClassifier` |
| Image Embeddings | `ClipImageEncoder` | `inference4j/clip-vit-base-patch32` | ~340 MB | `ImageEmbedder` |
| Text Embeddings (CLIP) | `ClipTextEncoder` | `inference4j/clip-vit-base-patch32` | ~255 MB | `TextEmbedder` |

## Audio

| Capability | Wrapper | Default Model ID | Size | API |
|------------|---------|-------------------|------|-----|
| Speech-to-Text | `Wav2Vec2Recognizer` | `inference4j/wav2vec2-base-960h` | ~370 MB | `SpeechRecognizer` |
| Voice Activity Detection | `SileroVadDetector` | `inference4j/silero-vad` | ~2 MB | `VoiceActivityDetector` |

## Generative AI

Generative models use a separate module (`inference4j-genai`) and a different builder pattern. See [Generative AI](../generative-ai/index.md) for details.

| Capability | Wrapper | Model ID | Size | License |
|------------|---------|----------|------|---------|
| Text Generation | `TextGenerator` | `inference4j/phi-3-mini-4k-instruct` | ~2.7 GB | MIT |
| Text Generation | `TextGenerator` | `inference4j/deepseek-r1-distill-qwen-1.5b` | ~1 GB | MIT |

## Model compatibility

### YOLOv8 / YOLO11

`YoloV8Detector` is compatible with both YOLOv8 and YOLO11 models — they share the same output layout (`[1, 4+C, N]`). It is **not** compatible with YOLOv5 (different layout with objectness column) or YOLO26 (NMS-free architecture).

### EfficientNet variants

`EfficientNetClassifier` is tested against EfficientNet-Lite4 (TensorFlow origin, softmax built-in). For PyTorch-exported EfficientNet models that output raw logits, override with `.outputOperator(OutputOperator.softmax())`.

### Custom models

Any ONNX-exported model works with the appropriate wrapper, as long as it follows the expected input/output layout. See the [Custom Models guide](../guides/model-loading.md) for details.

## Cache

Models are cached in `~/.cache/inference4j/` by default. Customize with:

- System property: `-Dinference4j.cache.dir=/path/to/cache`
- Environment variable: `INFERENCE4J_CACHE_DIR=/path/to/cache`

See [Configuration](configuration.md) for all options.

## Planned models

| Domain | Model | Status |
|--------|-------|--------|
| Text | TrOCR (text recognition) | Planned |
| Audio | Whisper (autoregressive STT) | Future |

See the [Roadmap](../roadmap.md) for details.
