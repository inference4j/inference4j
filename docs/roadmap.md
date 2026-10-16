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

### v0.13.0 — Long input, Whisper & readable transcripts

**Long input, processed in full or rejected — never silently cut**

- [x] **`TruncationPolicy`** — every token-limited wrapper either truncates with a warning
      (default) or fails with `InputTooLongException`; generators use the model's real
      position limit instead of a fixed 512
- [x] **`TokenWindows`** and **`.stride(int)`** — overlapping windows for NER and punctuation
      (lossless), the embedder (weighted average) and the reranker (MaxP)
- [x] **`WindowAggregation`** — the text classifier combines window scores with `mean()`,
      `max()` or a custom lambda
- [x] **`WordPieceTokenizer` parity with HuggingFace** — ASCII symbols split, accents
      stripped for uncased models, control characters dropped
- [x] **Prefill memory** — generation no longer copies `[prompt × vocabulary]` logits into
      the heap

**Speech**

- [x] **Whisper end to end** — `inference4j/whisper-tiny-genai` and `whisper-small-genai`
      exported with onnxruntime-genai's model builder; `WhisperSpeechModel` is a
      `SpeechRecognizer`
- [x] **`SegmentedSpeechRecognizer`** — cuts long audio at voice-activity pauses; whisper-small
      word error rate on TED-LIUM long-form drops from 11.7% to 3.7%
- [x] **Punctuation, casing & sentence segmentation** — `PunctCapSegModel`, plus
      **`PunctuatedSpeechRecognizer`** to make Wav2Vec2 transcripts readable
- [x] **`SileroVadDetector.speechPadding`** — keeps word edges from being clipped

**Platform**

- [x] **ONNX Runtime 1.28.0** and **onnxruntime-genai 0.17.0** — the official genai Java
      release, redistributed on Maven Central; one native ONNX Runtime on the classpath
- [x] **onnxruntime-genai is the default engine** for LLMs, vision-language and speech
- [x] **Fixes** — genai wrappers no longer repeat the last token; `UnigramTokenizer` maps
      unknown characters to `unk`; boolean tensors supported

### v0.13.1 — Generation fixes & newer LLMs

- [x] **Chat prompts keep their end** — over-long prompts keep the chat template and the end of the
      message, so the model still answers instead of continuing the text
      ([#76](https://github.com/inference4j/inference4j/issues/76))
- [x] **Qwen2.5 chat-template tokens** — special tokens are read from `tokenizer_config.json`, so the
      template is no longer tokenized as plain text
- [x] **KV cache stays native** — `NativeTensors` and `InferenceSession.runNative`; heap per generated token
      no longer grows with context ([#99](https://github.com/inference4j/inference4j/issues/99))
- [x] **Spring Boot auto-configuration for the genai wrappers** — `TextGenerator`, `VisionLanguageModel`,
      `WhisperSpeechModel` ([#90](https://github.com/inference4j/inference4j/issues/90))
- [x] **Newer genai LLMs** — Qwen3-0.6B, SmolLM3-3B and Phi-4-mini presets
      ([#87](https://github.com/inference4j/inference4j/issues/87))

## Planned — 0.14.0

Tracked in the [0.14.0 milestone](https://github.com/inference4j/inference4j/milestone/2).

### Zero-shot NLP, guardrails and RAG

- [ ] **Zero-shot NER** — GLiNER: extract entity types given at call time, plus a PII-tuned model ([#101](https://github.com/inference4j/inference4j/issues/101))
- [x] **Prompt-injection detector** — `DebertaPromptInjectionDetector`, a DeBERTa-v3 guardrail classifier (Apache-2.0
      default; Llama Prompt Guard 2 through a user-supplied source, since it is gated). `UnigramTokenizer` now applies
      the `tokenizer.json` normalizer (SentencePiece `Precompiled`, `Strip`, `Replace`), matching HuggingFace token for
      token on DeBERTa-v3 and T5 ([#102](https://github.com/inference4j/inference4j/issues/102))
- [ ] **Zero-shot text classification** — NLI-based, labels chosen at call time ([#103](https://github.com/inference4j/inference4j/issues/103))
- [ ] **Modern embeddings** — nomic-embed-text-v1.5 (8K context), Qwen3-Embedding-0.6B, and Matryoshka
      `.dimensions(int)` ([#104](https://github.com/inference4j/inference4j/issues/104))
- [ ] **Modern reranker** — bge-reranker-v2-m3, multilingual and long-context ([#105](https://github.com/inference4j/inference4j/issues/105))
- [ ] **LangChain4j adapter** — `EmbeddingModel`, `ScoringModel`, audio document parser, PII redaction and an
      input guardrail ([#111](https://github.com/inference4j/inference4j/issues/111))
- [ ] **Long input for generators** — sentence chunking for translation, map-reduce for summarization ([#74](https://github.com/inference4j/inference4j/issues/74))
- [x] **Encoder-decoder KV cache stays native** — FlanT5, BART, MarianMT, CoEdIT and T5SqlGenerator keep both
      caches in ONNX Runtime memory; heap per generated token drops from 2–21 MB to under 2 MB
      ([#119](https://github.com/inference4j/inference4j/issues/119))

### Speech

- [x] **Moonshine v2 speech-to-text** — `MoonshineRecognizer` (tiny, small, medium; English) with no fixed
      30-second window; works with `SegmentedSpeechRecognizer` ([#109](https://github.com/inference4j/inference4j/issues/109))
- [x] **Live transcription** — streaming Moonshine: feed audio as it arrives, get line-by-line text while the
      speaker talks ([#118](https://github.com/inference4j/inference4j/issues/118))
- [ ] **Kokoro TTS** — `SpeechSynthesizer` plus a permissively licensed Java phonemizer (espeak-ng is
      GPL-3.0); replaces Piper ([#50](https://github.com/inference4j/inference4j/issues/50))

### Vision & documents

- [x] **OCR recognizer and `OcrPipeline`** — PaddleOCR PP-OCRv6 (small and tiny) line detector and CTC recognizer,
      composed by `OcrPipeline`; `CraftTextDetector` fits as an alternative detector
      ([#106](https://github.com/inference4j/inference4j/issues/106)). TrOCR for handwriting follows
      ([#128](https://github.com/inference4j/inference4j/issues/128))
- [x] **CRAFT detects with its default thresholds** — post-processing now follows the reference implementation
      (max text score, separate link threshold, dilation, rotated boxes), matching it region for region
      ([#130](https://github.com/inference4j/inference4j/issues/130))
- [ ] **Apache-licensed detector** — RT-DETR / D-FINE, NMS-free, an alternative to the AGPL-3.0 Ultralytics
      models ([#107](https://github.com/inference4j/inference4j/issues/107))
- [ ] **Promptable segmentation** — SAM 2.1 tiny, with a `SegmentationMask` result type shared with
      semantic segmentation ([#108](https://github.com/inference4j/inference4j/issues/108))

### Generative AI via onnxruntime-genai

New LLM, vision-language and speech architectures come through `inference4j-genai` rather than being
re-implemented on the native engine. The native engine stays for encoder-decoder models (T5, BART, MarianMT)
and plain HuggingFace exports.

- [ ] **Qwen vision-language** — a second `VisionLanguageModel` backed by a Qwen VL export ([#88](https://github.com/inference4j/inference4j/issues/88))
- [ ] **Audio-in chat** — LFM2 / Phi-4-multimodal through `inference4j-genai` ([#89](https://github.com/inference4j/inference4j/issues/89))
- [ ] **License-restricted LLMs** — Llama-3.2 and Gemma-3 through a user-supplied source ([#110](https://github.com/inference4j/inference4j/issues/110))

## Unscheduled

- [ ] **Semantic segmentation** — SegFormer; reuses the `SegmentationMask` type from [#108](https://github.com/inference4j/inference4j/issues/108)
- [ ] **Super-resolution** — Real-ESRGAN. Needs denormalization before `TensorImages.toRgb`
- [ ] **Qwen3-Reranker** — decoder-token scoring needs a new `OutputOperator`; see [#105](https://github.com/inference4j/inference4j/issues/105) for the
      cross-encoder reranker
- [ ] **Tiktoken tokenizer** — deferred. No small, ONNX-viable model currently requires it
- [ ] **CRAFT improvements** — vertical text and mixed orientation
- [ ] **Stable Diffusion** — feasibility study of text-to-image on ONNX Runtime
- [ ] **More ViT models** — additional Vision Transformer variants

## Dropped

- ~~Generic Pipeline API~~ — `Pipeline.builder().stage().stage().build()` adds abstraction without value. Models are too different for a generic composition framework. Named pipelines (e.g., `OcrPipeline`) as concrete classes instead.
- ~~Native Whisper~~ — would require mel spectrogram / FFT preprocessing in Java to duplicate
  what onnxruntime-genai already does natively. Whisper ships through `inference4j-genai`.
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
| Text | Tiktoken LLM | Deferred |
| Text | GLiNER (zero-shot NER, PII) | 0.14.0 |
| Text | Prompt-injection detector (DeBERTa-v3) | 0.14.0 |
| Text | Zero-shot classification (NLI) | 0.14.0 |
| Text | nomic-embed-text-v1.5 / Qwen3-Embedding | 0.14.0 |
| Text | bge-reranker-v2-m3 | 0.14.0 |
| Text | Qwen3, SmolLM3, Phi-4-mini (via inference4j-genai) | v0.13.1 |
| Text | Punctuation, casing & segmentation (punct-cap-seg) | v0.13.0 |
| Vision | ResNet | Done |
| Vision | EfficientNet | Done |
| Vision | YOLOv8 / YOLO11 | Done |
| Vision | YOLO26 | Done |
| Vision | CLIP (visual search, zero-shot classification) | Done |
| Vision | Phi-3.5 Vision (captioning, VQA) | Done |
| Vision | Additional ViT models | Beyond |
| Vision | PaddleOCR recognizer + OCR Pipeline | 0.14.0 |
| Vision | RT-DETR / D-FINE (Apache-licensed detection) | 0.14.0 |
| Vision | SAM 2.1 tiny (promptable segmentation) | 0.14.0 |
| Vision | Qwen VL (vision-language) | 0.14.0 |
| Vision | Stable Diffusion (text-to-image) | Beyond — feasibility study |
| Audio | Wav2Vec2-CTC (speech-to-text) | Done |
| Audio | Silero VAD (voice activity detection) | Done |
| Audio | Kokoro TTS (text-to-speech, replaces Piper) | 0.14.0 |
| Audio | Moonshine v2 (speech-to-text, live transcription) | Done |
| Audio | Whisper (autoregressive speech-to-text) | Done via inference4j-genai |
