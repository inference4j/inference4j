# Roadmap

## Completed

### Phase 1: Foundation (Core & NLP)

- [x] `inference4j-core` — `InferenceSession`, `Tensor`, `ModelSource`, `MathOps` (softmax, sigmoid, logSoftmax, topK, NMS, cxcywh2xyxy)
- [x] `inference4j-preprocessing` — `WordPieceTokenizer`, `EncodedInput`, `Tokenizer` interface
- [x] `SentenceTransformer` wrapper — sentence embeddings with CLS/MEAN/MAX pooling
- [x] `EmbeddingModelRouter` — A/B testing with round-robin routing

### Phase 2: Vision

- [x] Image preprocessing — `ImageTransformPipeline`, `ResizeTransform`, `CenterCropTransform`, `ImageLayout` (NCHW/NHWC), `Labels` (ImageNet/COCO presets)
- [x] `ResNet` wrapper — image classification with ImageNet defaults
- [x] `EfficientNet` wrapper — image classification with TensorFlow defaults

### Phase 2.5: Object Detection

- [x] `YoloV8` wrapper — NMS-based detection (also compatible with YOLO11)
- [x] `Yolo26` wrapper — NMS-free detection
- [x] `ObjectDetector` interface, `Detection`, `BoundingBox` result types

### Phase 3: Audio

- [x] `Wav2Vec2` wrapper — CTC speech-to-text (single-pass, non-autoregressive)
- [x] Audio preprocessing — `AudioLoader` (WAV loading), `AudioProcessor` (resample, normalize), `Vocabulary` (vocab.json)
- [x] `MathOps.ctcGreedyDecode()` — CTC greedy decoding
- [x] `SpeechRecognizer` interface, `Transcription` result type
- [x] Silero VAD wrapper — voice activity detection
- [x] Hardware acceleration benchmarks (CoreML: ResNet 3.7x, CRAFT 5.4x)

### Phase 3.5: NLP

- [x] `DistilBertTextClassifier` wrapper — text classification with auto-detection of softmax/sigmoid from `config.json`
- [x] `MiniLMSearchReranker` wrapper — cross-encoder query-document relevance scoring
- [x] `TextClassifier` and `SearchReranker` interfaces, `TextClassification` result type
- [x] `ModelConfig` — parses HuggingFace `config.json` for `id2label` and `problem_type`
- [x] Sentence pair encoding in `Tokenizer`/`WordPieceTokenizer`

### Architecture

- [x] `AbstractInferenceTask` — enforced preprocess → infer → postprocess pipeline with `final run()`
- [x] `Preprocessor`/`Postprocessor` functional interfaces
- [x] `InferenceContext` — cross-stage data carrier
- [x] Task-oriented architecture — `InferenceTask` → `Classifier`/`Detector` → domain interfaces
- [x] Builder API — `.session()` package-private, public API uses `modelId` + `modelSource` + `sessionOptions(SessionConfigurer)`
- [x] Spring Boot starter — auto-configuration, health indicators
- [x] Documentation site (MkDocs Material)
- [x] CRAFT text detection wrapper — `TextDetector` interface, `TextRegion`, `CraftTextDetector`

## Next Up

### CLIP — Visual Search & Zero-Shot Classification

[CLIP](https://openai.com/research/clip) (Contrastive Language–Image Pre-training) maps images and text into a shared embedding space. This unlocks two high-value use cases:

- **Visual search** — find images that match a text query, or find text that matches an image
- **Zero-shot classification** — classify images against arbitrary text labels without any training

CLIP is a single-pass model (no autoregressive decoding), so it fits naturally into inference4j's existing architecture.

- [ ] CLIP image encoder
- [ ] CLIP text encoder
- [ ] Combined `ClipModel` with `similarity(image, texts)` API
- [ ] Runnable example in `inference4j-examples`

### Model Test Suite

Integration tests that download real models and verify inference output end-to-end. Separate from unit tests so `./gradlew test` stays fast and offline.

- [ ] `./gradlew modelTest` Gradle task
- [ ] Coverage across all supported model wrappers
- [ ] CI integration on a schedule (not on every PR)

## Parked

### Autoregressive Generation

TrOCR, Whisper, and any decoder-based model require an autoregressive generate loop — token-by-token decoding with KV cache management. This is fundamentally different from the single-pass pipeline used by all current models, and requires significant infrastructure:

- Generate loop with configurable stopping criteria
- KV cache management
- BPE tokenizer
- Mel spectrogram / FFT (for Whisper)

**Blocked models:** TrOCR (text recognition), Whisper (speech-to-text), OCR Pipeline (depends on TrOCR).

We'll revisit once CLIP and the model test suite are complete.

## Dropped

- ~~Generic Pipeline API~~ — `Pipeline.builder().stage().stage().build()` adds abstraction without value. Models are too different for a generic composition framework. Named pipelines (e.g., `OcrPipeline`) as concrete classes instead.
- ~~Codegen plugin~~ — generates type-safe wrappers from `.onnx` metadata but doesn't solve preprocessing/postprocessing, which is where the real complexity lives. Handcrafted wrappers deliver more value.

## Target models

| Domain | Model | Status |
|--------|-------|--------|
| Text | SentenceTransformer (all-MiniLM, all-mpnet, BERT) | Done |
| Text | Cross-encoder reranker (ms-marco-MiniLM) | Done |
| Text | Text classification (DistilBERT, sentiment, moderation) | Done |
| Text | CRAFT (text detection) | Done |
| Text | TrOCR (text recognition) | Parked |
| Vision | ResNet | Done |
| Vision | EfficientNet | Done |
| Vision | YOLOv8 / YOLO11 | Done |
| Vision | YOLO26 | Done |
| Vision | CLIP (visual search) | Next |
| Audio | Wav2Vec2-CTC (speech-to-text) | Done |
| Audio | Silero VAD (voice activity detection) | Done |
| Audio | Whisper (autoregressive speech-to-text) | Parked |
