# inference4j Roadmap

## Phase 1: Foundation (Core & NLP) — Done
- [x] `inference4j-core` — `InferenceSession`, `Tensor`, `ModelSource`, `MathOps` (softmax, sigmoid, logSoftmax, topK, NMS, cxcywh2xyxy)
- [x] `inference4j-preprocessing` — `WordPieceTokenizer`, `EncodedInput`, `Tokenizer` interface
- [x] `SentenceTransformer` wrapper — sentence embeddings with CLS/MEAN/MAX pooling
- [x] `EmbeddingModelRouter` — A/B testing with round-robin routing

## Phase 2: Vision — Done
- [x] Image preprocessing — `ImageTransformPipeline`, `ResizeTransform`, `CenterCropTransform`, `ImageLayout` (NCHW/NHWC), `Labels` (ImageNet/COCO presets)
- [x] `ResNet` wrapper — image classification with ImageNet defaults
- [x] `EfficientNet` wrapper — image classification with TensorFlow defaults

## Phase 2.5: Object Detection — Done
- [x] `YoloV8` wrapper — NMS-based detection (also compatible with YOLO11)
- [x] `Yolo26` wrapper — NMS-free detection
- [x] `ObjectDetectionModel` interface, `Detection`, `BoundingBox` result types

## Phase 3: Audio — In Progress
- [x] `Wav2Vec2` wrapper — CTC speech-to-text (single-pass, non-autoregressive)
- [x] Audio preprocessing — `AudioLoader` (WAV loading), `AudioProcessor` (resample, normalize), `Vocabulary` (vocab.json)
- [x] `MathOps.ctcGreedyDecode()` — CTC greedy decoding
- [x] `SpeechToTextModel` interface, `Transcription` result type
- [x] Silero VAD wrapper — voice activity detection
- [ ] Benchmarks — latency and throughput measurements

## Phase 3.5: NLP — Done
- [x] `DistilBertClassifier` wrapper — text classification (sentiment, moderation, intent detection) with auto-detection of softmax/sigmoid from `config.json`
- [x] `MiniLMReranker` wrapper — cross-encoder query-document relevance scoring, completes the search/RAG pipeline alongside `SentenceTransformer`
- [x] `TextClassificationModel` and `CrossEncoderModel` interfaces, `TextClassification` result type
- [x] `ModelConfig` — parses HuggingFace `config.json` for `id2label` and `problem_type` (via Jackson)
- [x] Sentence pair encoding in `Tokenizer`/`WordPieceTokenizer` — `[CLS] textA [SEP] textB [SEP]` with segment IDs

## Phase 4: Pipelines
- [x] CRAFT text detection wrapper — `TextDetectionModel` interface, `TextRegion`, `Craft` wrapper
- [ ] Pipeline API — `Pipeline.builder().stage("name", model).build()`
- [ ] OCR Pipeline — TrOCR recognition + embedding-based correction (CRAFT detection done)
- [ ] Vision Pipeline — detect + classify composition

## Phase 5: Ecosystem
- [ ] Documentation site
- [ ] HuggingFace Model Hub integration (`ModelSource` implementation)
- [ ] CLIP visual search wrapper

## Phase 6: Codegen
- [ ] Maven/Gradle plugin for type-safe wrapper generation from `.onnx` metadata
- [ ] Generate I/O POJOs and Model classes

## Target Models

| Domain | Model | Status |
|--------|-------|--------|
| Text | SentenceTransformer (all-MiniLM, all-mpnet, BERT) | Done |
| Text | Cross-encoder reranker (ms-marco-MiniLM) | Done |
| Text | Text classification (DistilBERT, sentiment, moderation) | Done |
| Text | CRAFT (text detection) | Done |
| Text | TrOCR/EasyOCR (text recognition) | Planned (Phase 4) |
| Vision | ResNet | Done |
| Vision | EfficientNet | Done |
| Vision | YOLOv8 / YOLO11 | Done |
| Vision | YOLO26 | Done |
| Vision | CLIP (visual search) | Planned (Phase 5) |
| Vision | MobileNet | Planned |
| Audio | Wav2Vec2-CTC (speech-to-text) | Done |
| Audio | Silero VAD (voice activity detection) | Done |
| Audio | Whisper (autoregressive speech-to-text) | Future |
