# Supported Models

All supported models are hosted under the [`inference4j`](https://huggingface.co/inference4j) HuggingFace organization and are automatically downloaded and cached on first use.

## NLP

| Capability | Wrapper | Default Model ID | Size | API |
|------------|---------|-------------------|------|-----|
| Text Classification | `DistilBertTextClassifier` | `inference4j/distilbert-base-uncased-finetuned-sst-2-english` | ~260 MB | `TextClassifier` |
| Text Embeddings | `SentenceTransformerEmbedder` | `inference4j/all-MiniLM-L6-v2` | ~90 MB | `TextEmbedder` |
| Search Reranking | `MiniLMSearchReranker` | `inference4j/ms-marco-MiniLM-L-6-v2` | ~90 MB | `SearchReranker` |
| Text Generation | `OnnxTextGenerator.gpt2()` | `inference4j/gpt2` | ~500 MB | `TextGenerator` |
| Text Generation | `OnnxTextGenerator.smolLM2()` | `inference4j/smollm2-360m-instruct` | ~700 MB | `TextGenerator` |
| Text Generation | `OnnxTextGenerator.smolLM2_1_7B()` | `inference4j/smollm2-1.7b-instruct` | ~3.4 GB | `TextGenerator` |
| Text Generation | `OnnxTextGenerator.tinyLlama()` | `inference4j/tinyllama-1.1b-chat` | ~2.2 GB | `TextGenerator` |
| Text Generation | `OnnxTextGenerator.qwen2()` | `inference4j/qwen2.5-1.5b-instruct` | ~3 GB | `TextGenerator` |
| Text Generation | `OnnxTextGenerator.gemma2()` | Gated — requires manual download | ~5 GB | `TextGenerator` |
| Summarization / Translation / Grammar | `FlanT5TextGenerator.flanT5Small()` | `inference4j/flan-t5-small` | ~300 MB | `TextGenerator`, `Summarizer`, `Translator`, `GrammarCorrector` |
| Summarization / Translation / Grammar | `FlanT5TextGenerator.flanT5Base()` | `inference4j/flan-t5-base` | ~900 MB | `TextGenerator`, `Summarizer`, `Translator`, `GrammarCorrector` |
| Summarization / Translation / Grammar | `FlanT5TextGenerator.flanT5Large()` | `inference4j/flan-t5-large` | ~3 GB | `TextGenerator`, `Summarizer`, `Translator`, `GrammarCorrector` |
| Text-to-SQL | `T5SqlGenerator.t5SmallAwesome()` | `inference4j/t5-small-awesome-text-to-sql` | ~240 MB | `TextGenerator`, `SqlGenerator` |
| Text-to-SQL | `T5SqlGenerator.t5LargeSpider()` | `inference4j/T5-LM-Large-text2sql-spider` | ~4.6 GB | `TextGenerator`, `SqlGenerator` |
| Summarization | `BartSummarizer.distilBartCnn()` | `inference4j/distilbart-cnn-12-6` | ~1.2 GB | `TextGenerator`, `Summarizer` |
| Summarization | `BartSummarizer.bartLargeCnn()` | `inference4j/bart-large-cnn` | ~1.6 GB | `TextGenerator`, `Summarizer` |
| Translation | `MarianTranslator.builder()` | User-specified (`inference4j/opus-mt-*`) | varies | `TextGenerator`, `Translator` |
| Grammar Correction | `CoeditGrammarCorrector.coeditBase()` | `inference4j/coedit-base` | ~900 MB | `TextGenerator`, `GrammarCorrector` |
| Grammar Correction | `CoeditGrammarCorrector.coeditLarge()` | `inference4j/coedit-large` | ~3 GB | `TextGenerator`, `GrammarCorrector` |

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

Generative models use a separate module (`inference4j-genai`) and a different builder pattern. See [Generative AI](../generative-ai/introduction.md) for details.

| Capability | Wrapper | Model ID | Size | License |
|------------|---------|----------|------|---------|
| Text Generation | `TextGenerator` | `inference4j/phi-3-mini-4k-instruct` | ~2.7 GB | MIT |
| Text Generation | `TextGenerator` | `inference4j/deepseek-r1-distill-qwen-1.5b` | ~1 GB | MIT |
| Speech-to-Text / Translation (WIP) | `WhisperSpeechModel` | `inference4j/whisper-small-genai` | ~500 MB | MIT |
| Vision-Language | `VisionLanguageModel` | `inference4j/phi-3.5-vision-instruct` | ~3.3 GB | MIT |

## Model reference

A comprehensive view of all supported models, organized by architecture:

### Encoder-only (single-pass)

| Model | Tokenizer | Wrapper | Use Cases |
|-------|-----------|---------|-----------|
| DistilBERT SST-2 | WordPiece | `DistilBertTextClassifier` | Sentiment analysis, text classification |
| all-MiniLM-L6-v2 | WordPiece | `SentenceTransformerEmbedder` | Semantic search, embeddings |
| MiniLM-L-6 MS MARCO | WordPiece | `MiniLMSearchReranker` | Search reranking |

### Decoder-only (autoregressive)

| Model | Tokenizer | Wrapper | Use Cases |
|-------|-----------|---------|-----------|
| GPT-2 | BPE | `OnnxTextGenerator` | Text completion |
| SmolLM2-360M | BPE | `OnnxTextGenerator` | Chat, instruction following |
| TinyLlama-1.1B | SentencePiece BPE | `OnnxTextGenerator` | Chat, instruction following |
| Qwen2.5-1.5B | BPE | `OnnxTextGenerator` | Chat, instruction following |
| Gemma 2-2B | SentencePiece BPE | `OnnxTextGenerator` | Chat, instruction following |

### Encoder-decoder (autoregressive)

| Model | Tokenizer | Wrapper | Use Cases |
|-------|-----------|---------|-----------|
| Flan-T5 (Small / Base / Large) | SentencePiece Unigram | `FlanT5TextGenerator` | Summarization, translation, grammar |
| T5-small-awesome-text-to-sql | SentencePiece Unigram | `T5SqlGenerator` | Text-to-SQL (lightweight) |
| T5-LM-Large-text2sql-spider | SentencePiece Unigram | `T5SqlGenerator` | Text-to-SQL (high accuracy) |
| DistilBART CNN 12-6 | BPE | `BartSummarizer` | Summarization |
| BART Large CNN | BPE | `BartSummarizer` | Summarization |
| MarianMT (opus-mt-*) | SentencePiece BPE | `MarianTranslator` | Translation (fixed language pair) |
| CoEdIT (Base / Large) | SentencePiece Unigram | `CoeditGrammarCorrector` | Grammar correction |

### Vision

| Model | Tokenizer | Wrapper | Use Cases |
|-------|-----------|---------|-----------|
| ResNet-50 | N/A | `ResNetClassifier` | Image classification |
| EfficientNet-Lite4 | N/A | `EfficientNetClassifier` | Image classification |
| YOLOv8n | N/A | `YoloV8Detector` | Object detection |
| YOLO26n | N/A | `Yolo26Detector` | Object detection |
| CRAFT | N/A | `CraftTextDetector` | Text detection in images |

### Multimodal

| Model | Tokenizer | Wrapper | Use Cases |
|-------|-----------|---------|-----------|
| CLIP ViT-B/32 | BPE | `ClipClassifier` | Zero-shot image classification |

### Audio

| Model | Tokenizer | Wrapper | Use Cases |
|-------|-----------|---------|-----------|
| Wav2Vec2 | CTC | `Wav2Vec2Recognizer` | Speech-to-text |
| Silero VAD | N/A | `SileroVadDetector` | Voice activity detection |

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
| Text | TrOCR (text recognition) | Planned — enabled by encoder-decoder infrastructure |

See the [Roadmap](../roadmap.md) for details.
