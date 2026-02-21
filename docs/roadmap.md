# Roadmap

## Completed

### Phase 1: Foundation (Core & NLP)

- [x] `inference4j-core` — `InferenceSession`, `Tensor`, `ModelSource`, `MathOps` (softmax, sigmoid, logSoftmax, topK, NMS, cxcywh2xyxy)
- [x] Tokenizers — `WordPieceTokenizer`, `BpeTokenizer`, `DecodingBpeTokenizer`, `EncodedInput`, `Tokenizer` interface
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

### Phase 4: CLIP — Visual Search & Zero-Shot Classification

- [x] CLIP image encoder and text encoder
- [x] `ClipClassifier` — zero-shot image classification against arbitrary text labels
- [x] `ClipModel` with `similarity(image, texts)` API
- [x] `BpeTokenizer` — byte-level BPE for CLIP/GPT-2 family
- [x] Runnable examples in `inference4j-examples`

### Phase 5: Autoregressive Generation

- [x] `GenerationEngine` — pure ONNX Runtime autoregressive loop with KV cache
- [x] `GenerativeTask` / `GenerativeSession` — generation contracts in core
- [x] Sampling pipeline — `LogitsProcessor`, `GreedySampler`, `CategoricalSampler`, temperature/topK/topP
- [x] `TokenStreamer` — streaming token delivery with stop sequence support
- [x] `Gpt2TextGenerator` — text generation with GPT-2 (pure ONNX Runtime, no genai dependency)
- [x] `DecodingBpeTokenizer` / `TokenDecoder` — BPE tokenizer with decoding support
- [x] `inference4j-genai` — onnxruntime-genai backed generation for larger models (Phi-3, DeepSeek-R1, Phi-3.5 Vision)
- [x] Streaming generation API — token-by-token callbacks via `Consumer<String>`

### Architecture & Ecosystem

- [x] `AbstractInferenceTask` — enforced preprocess → infer → postprocess pipeline with `final run()`
- [x] `Preprocessor`/`Postprocessor` functional interfaces
- [x] `InferenceContext` — cross-stage data carrier
- [x] Task-oriented architecture — `InferenceTask` → `Classifier`/`Detector` → domain interfaces
- [x] Builder API — `.session()` package-private, public API uses `modelId` + `modelSource` + `sessionOptions(SessionConfigurer)`
- [x] Spring Boot starter — auto-configuration, health indicators
- [x] Documentation site (MkDocs Material)
- [x] CRAFT text detection wrapper — `TextDetector` interface, `TextRegion`, `CraftTextDetector`
- [x] Model test suite — `./gradlew modelTest` with real model downloads and inference verification
- [x] Module consolidation — `inference4j-tasks` and `inference4j-preprocessing` merged into `inference4j-core`

## Next Up

- [ ] **TrOCR** — text recognition (handwriting, printed text)
- [ ] OCR Pipeline — CRAFT detection + TrOCR recognition composed end-to-end
- [ ] **Whisper** — autoregressive speech-to-text via pure ONNX Runtime (mel spectrogram / FFT preprocessing)
- [ ] Mel spectrogram / FFT preprocessing (unlocks Whisper)

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
| Text | TrOCR (text recognition) | Next |
| Text | GPT-2 (text generation) | Done |
| Vision | ResNet | Done |
| Vision | EfficientNet | Done |
| Vision | YOLOv8 / YOLO11 | Done |
| Vision | YOLO26 | Done |
| Vision | CLIP (visual search, zero-shot classification) | Done |
| Vision | Phi-3.5 Vision (captioning, VQA) | Done |
| Audio | Wav2Vec2-CTC (speech-to-text) | Done |
| Audio | Silero VAD (voice activity detection) | Done |
| Audio | Whisper (autoregressive speech-to-text) | Next |
