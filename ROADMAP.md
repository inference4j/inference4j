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
- [ ] Silero VAD wrapper — voice activity detection
- [ ] Benchmarks — latency and throughput measurements

## Phase 4: Pipelines
- [ ] Pipeline API — `Pipeline.builder().stage("name", model).build()`
- [ ] OCR Pipeline — CRAFT text detection + TrOCR recognition + embedding-based correction
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
| Text | CRAFT (text detection) | Planned (Phase 4) |
| Text | TrOCR/EasyOCR (text recognition) | Planned (Phase 4) |
| Vision | ResNet | Done |
| Vision | EfficientNet | Done |
| Vision | YOLOv8 / YOLO11 | Done |
| Vision | YOLO26 | Done |
| Vision | CLIP (visual search) | Planned (Phase 5) |
| Vision | MobileNet | Planned |
| Audio | Wav2Vec2-CTC (speech-to-text) | Done |
| Audio | Silero VAD (voice activity detection) | Planned (Phase 3) |
| Audio | Whisper (autoregressive speech-to-text) | Future |
