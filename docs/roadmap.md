# Roadmap

## Completed

### Phase 1: Foundation (Core & NLP)

- [x] `inference4j-core` — `InferenceSession`, `Tensor`, `ModelSource`, `MathOps` (softmax, sigmoid, logSoftmax, topK, NMS, cxcywh2xyxy)
- [x] `inference4j-preprocessing` — `WordPieceTokenizer`, `BpeTokenizer`, `EncodedInput`, `Tokenizer` interface
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

## Next Up

### Autoregressive Generation via onnxruntime-genai

All models in inference4j today are **single-pass** — one forward pass, one result. A large class of models require **autoregressive generation** instead: producing output token-by-token, where each token depends on all previous tokens. This includes language models, speech-to-text with attention, and vision-language models.

[onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) provides exactly this capability — a generate loop with KV cache management, sampling strategies, and streaming — built on top of ONNX Runtime. We've published [community Java bindings](https://github.com/inference4j/onnxruntime-genai) to Maven Central (`io.github.inference4j:onnxruntime-genai`) since Microsoft does not currently publish them.

This is the next major focus area, and it unlocks an entirely new category of models:

- [ ] Integrate `onnxruntime-genai` Java bindings into inference4j
- [ ] **Whisper** — autoregressive speech-to-text with attention (replaces CTC-based Wav2Vec2 for multilingual/high-accuracy use cases)
- [ ] **GPT-2** — text generation
- [ ] **Phi-3** — small language model for local inference
- [ ] **TrOCR** — text recognition (handwriting, printed text)
- [ ] **ViT + decoder models** — vision-language tasks (image captioning, visual Q&A)
- [ ] OCR Pipeline — CRAFT detection + TrOCR recognition composed end-to-end
- [ ] Mel spectrogram / FFT preprocessing (for Whisper)
- [ ] Streaming generation API — token-by-token callbacks

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
| Text | GPT-2 (text generation) | Next |
| Vision | ResNet | Done |
| Vision | EfficientNet | Done |
| Vision | YOLOv8 / YOLO11 | Done |
| Vision | YOLO26 | Done |
| Vision | CLIP (visual search, zero-shot classification) | Done |
| Vision | Phi-3 Vision / ViT-decoder (captioning, VQA) | Next |
| Audio | Wav2Vec2-CTC (speech-to-text) | Done |
| Audio | Silero VAD (voice activity detection) | Done |
| Audio | Whisper (autoregressive speech-to-text) | Next |
