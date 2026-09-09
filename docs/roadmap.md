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
- [x] `OnnxTextGenerator` — native text generation with GPT-2, SmolLM2, Qwen2.5 (pure ONNX Runtime, no genai dependency)
- [x] `DecodingBpeTokenizer` / `TokenDecoder` — BPE tokenizer with decoding support
- [x] `inference4j-genai` — onnxruntime-genai backed generation for larger models (Phi-3, DeepSeek-R1, Phi-3.5 Vision)
- [x] Streaming generation API — token-by-token callbacks via `Consumer<String>`

### Phase 6: Encoder-Decoder Generation

- [x] `EncoderDecoderSession` — encoder-decoder autoregressive loop with cross-attention and self-attention KV caches
- [x] `AbstractEncoderDecoderBuilder` — shared builder for encoder-decoder wrappers
- [x] `FlanT5TextGenerator` — multi-task text generation (summarization, translation, SQL, grammar) with Flan-T5
- [x] `BartSummarizer` — text summarization with BART / DistilBART
- [x] `MarianTranslator` — machine translation with MarianMT (Helsinki-NLP opus-mt models)
- [x] `CoeditGrammarCorrector` — grammar correction with CoEdIT
- [x] Task interfaces — `TextGenerator`, `Summarizer`, `Translator`, `GrammarCorrector`, `SqlGenerator`
- [x] `Language` enum — 24 languages for typed translation APIs

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

### v0.10.0 — NER & Embeddings

- [x] **Named Entity Recognition** — `BertNerRecognizer` with IOB2 tagging, cased WordPiece tokenizer, subword-to-word alignment via wordIds
- [x] `NamedEntityRecognizer` interface, `NamedEntity` result type (entity text, label, character offsets, confidence)
- [x] **Improved embeddings** — L2 normalization (`.normalize()`), text prefix (`.textPrefix()`), support for BGE, GTE, mpnet models
- [x] **Cased tokenizer** — `WordPieceTokenizer.fromVocabFile(path, lowercase)` for cased models
- [x] Spring Boot auto-configuration for `NamedEntityRecognizer`

### v0.10.1 — Dependency refresh

- [x] **onnxruntime-genai 0.12.0 &rarr; 0.15.2** — refreshed our
      [shadow build](https://github.com/inference4j/onnxruntime-genai); picks up upstream
      security fixes and Gemma 4 / int8 support
- [x] **ONNX Runtime 1.23.0 &rarr; 1.26.0** — matches the version the genai natives are
      built against, so both modules load the same native library
- [x] Documented the ONNX Runtime version contract between `inference4j-core` and
      `inference4j-genai`

### v0.11.0 — Spring Boot 4

- [x] **Spring Boot 4 starter** — the starter now targets Spring Boot 4.0+ / Spring Framework 7
- [x] Actuator health types moved to `org.springframework.boot.health.contributor`
- [x] [Migration guide](guides/spring-boot.md#migrating-from-010x) for existing users

Spring Boot 3.x left open-source support on 30 June 2026, and Spring AI 2.0 (GA June 2026)
requires Boot 4 and cannot load in a 3.x context. Boot 4 is also binary-incompatible with
Boot 3 for the actuator health types the starter uses, so a single artifact cannot serve
both — 0.10.1 is the final Boot 3 release and remains on Maven Central, frozen.

Only the starter is affected. `inference4j-core` and every other module have no Spring
dependency.

### v0.12.0 — Depth Estimation

- [x] **`Tensor.toFloats3D()`** — strict 3D reshape mirroring `toFloats2D()`, composing with
      `squeeze()` for batched model output
- [x] **`TensorImages`** — dense model output to `BufferedImage`: grayscale, RGB from CHW
      planes, and colormapped rendering
- [x] **`Colormap`** — grayscale, viridis and turbo ramps for visualizing dense output
- [x] **`MathOps.minMax` / `minMaxNormalize`** — the spatial helpers dense output needs
- [x] **Depth estimation** — `DepthEstimator`, `DepthMap`, and `DepthAnythingEstimator`
      (Depth Anything V2 Small)

The first four are the shared foundation for pixel-level tasks that `CLAUDE.md` had flagged
as a prerequisite. Depth estimation is its first consumer; semantic segmentation and
super-resolution are unblocked behind it.

## Next Up

### Tokenizers & LLMs

- [ ] **Tiktoken tokenizer** — deferred. No small, ONNX-viable model currently requires it;
      the realistic near-term targets use byte-level BPE or SentencePiece Unigram, both of
      which are already implemented. Revisit when a concrete model needs it.

### Speech

- [ ] **Moonshine speech-to-text** — raw-waveform ASR (MIT, 27M/62M) that performs feature
      extraction inside the ONNX graph, so it needs no mel-spectrogram/FFT work in Java.
      Reuses the existing raw-waveform audio pipeline.

### Embeddings & reranking

- [ ] **Qwen3-Embedding / Qwen3-Reranker (0.6B)** — Apache 2.0 with existing ONNX exports.
      Uses byte-level BPE we already support; the reranker scores via decoder tokens rather
      than a cross-encoder head, so it needs a new `OutputOperator`.

### Text-to-Speech

- [ ] **Kokoro TTS** — replaces Piper as the TTS target. Apache 2.0 end to end, versus
      Piper's development having moved to a GPL-3.0 fork.
- [ ] **Java phonemizer** — Kokoro needs grapheme-to-phoneme conversion, and the usual
      fallback (espeak-ng) is GPL-3.0. Needs a dictionary-based phonemizer over a
      permissively licensed lexicon. This is the real cost of the TTS milestone.
- [ ] `SpeechSynthesizer` interface, audio output generation

### Pixel-level tasks

The shared foundation shipped in v0.12.0, so both of these are now wrapper-level work.

- [ ] **Semantic segmentation** — SegFormer. Needs a `SegmentationMask` result type and
      per-pixel argmax; `CraftTextDetector.connectedComponents` is a promotion candidate
- [ ] **Super-resolution** — Real-ESRGAN. Needs denormalization before `TensorImages.toRgb`

### Beyond

- [ ] **OCR Pipeline** — CRAFT detection + TrOCR recognition composed end-to-end; study viability of full TrOCR models
- [ ] **CRAFT improvements** — test and improve support for vertical text and mixed orientation
- [ ] **Whisper** — study cost of mel spectrogram / FFT preprocessing; native autoregressive speech-to-text
- [ ] **Stable Diffusion** — study feasibility of text-to-image models on ONNX Runtime, including lightweight variants with GPU acceleration
- [ ] **More ViT models** — additional Vision Transformer variants (low effort, reuse existing image classification infrastructure)

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
| Text | GPT-2 (text generation) | Done |
| Text | SmolLM2-360M-Instruct (text generation) | Done |
| Text | Qwen2.5-1.5B-Instruct (text generation) | Done |
| Text | Flan-T5 (summarization, translation, SQL, grammar) | Done |
| Text | BART / DistilBART (summarization) | Done |
| Text | MarianMT (translation) | Done |
| Text | CoEdIT (grammar correction) | Done |
| Text | BERT NER (named entity recognition) | v0.10.0 |
| Text | BGE / GTE / E5 (improved embeddings) | v0.10.0 |
| Text | Tiktoken LLM | v0.11.0 |
| Vision | ResNet | Done |
| Vision | EfficientNet | Done |
| Vision | YOLOv8 / YOLO11 | Done |
| Vision | YOLO26 | Done |
| Vision | CLIP (visual search, zero-shot classification) | Done |
| Vision | Phi-3.5 Vision (captioning, VQA) | Done |
| Vision | Additional ViT models | Beyond |
| Vision | TrOCR + OCR Pipeline | Beyond |
| Vision | Stable Diffusion (text-to-image) | Beyond — feasibility study |
| Audio | Wav2Vec2-CTC (speech-to-text) | Done |
| Audio | Silero VAD (voice activity detection) | Done |
| Audio | Piper TTS (text-to-speech) | v0.12.0 |
| Audio | Whisper (autoregressive speech-to-text) | Beyond — feasibility study |
