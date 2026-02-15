# Visual Search

Classify images using arbitrary text labels — no training required — powered by [CLIP](https://arxiv.org/abs/2103.00020).

## Quick start

```java
try (ClipClassifier classifier = ClipClassifier.builder()
        .labels("cat", "dog", "bird", "car", "airplane")
        .build()) {
    List<Classification> results = classifier.classify(Path.of("photo.jpg"));
    System.out.println(results.get(0).label());      // "cat"
    System.out.println(results.get(0).confidence());  // 0.92
}
```

## Zero-shot classification

Unlike traditional image classifiers that are trained on a fixed set of labels, CLIP classifies images against **any labels you provide at runtime**. Just pass the labels you need — no retraining, no fine-tuning:

```java
// Emotion detection
ClipClassifier emotionClassifier = ClipClassifier.builder()
        .labels("happy", "sad", "angry", "surprised", "neutral")
        .promptTemplate("a photo of a {} person")
        .build();

// Product categorization
ClipClassifier productClassifier = ClipClassifier.builder()
        .labels("electronics", "clothing", "furniture", "food", "sports equipment")
        .promptTemplate("a product photo of {}")
        .build();

// Scene classification
ClipClassifier sceneClassifier = ClipClassifier.builder()
        .labels("beach", "mountain", "city", "forest", "desert")
        .promptTemplate("a landscape photo of a {}")
        .build();
```

## How it works

CLIP uses two separate encoders — one for images, one for text — trained so that matching image-text pairs produce similar embeddings. `ClipClassifier` wraps both encoders:

1. **At build time**: encodes each label into a text embedding using the prompt template (e.g. "a photo of a cat")
2. **At classify time**: encodes the image, computes similarity against all label embeddings, and returns ranked results with confidence scores

```
                                     ┌─ "a photo of a cat"  → [512-dim] ─┐
Labels (build time) → prompt template├─ "a photo of a dog"  → [512-dim] ─┤
                                     └─ "a photo of a bird" → [512-dim] ─┤
                                                                          ├─ similarity → softmax → Classification
Image (classify time) ────────────────────────────────────── [512-dim] ──┘
```

## Builder options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `labels(String...)` | `String[]` | (required) | The classification labels |
| `labels(List<String>)` | `List<String>` | (required) | The classification labels |
| `promptTemplate(String)` | `String` | `"a photo of a {}"` | Template for label encoding — `{}` is replaced with each label |
| `defaultTopK(int)` | `int` | Number of labels | How many results to return by default |
| `modelId(String)` | `String` | `inference4j/clip-vit-base-patch32` | HuggingFace model ID |
| `modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Where to load the model from |
| `sessionOptions(SessionConfigurer)` | `SessionConfigurer` | Default (CPU) | ONNX Runtime session options |

## Prompt template tips

The prompt template affects classification quality. CLIP was trained on natural language captions, so templates that resemble captions work best:

| Use case | Good template | Why |
|----------|---------------|-----|
| General objects | `"a photo of a {}"` | Default, works well for most cases |
| Fine-grained | `"a photo of a {}, a type of pet"` | Adds context for disambiguation |
| Scenes | `"a photo of a {}"` or `"a {} landscape"` | Matches CLIP training data |
| Actions | `"a photo of a person {}"` | e.g., "running", "swimming" |
| Styles | `"a {} style painting"` | e.g., "impressionist", "cubist" |

## Advanced: direct encoder access

For use cases beyond classification — image search, image-text similarity, or custom pipelines — use `ClipImageEncoder` and `ClipTextEncoder` directly. See the [CLIP Encoders reference](../reference/clip-encoders.md).

## Alternative models

The default model is `inference4j/clip-vit-base-patch32` (ViT-B/32) — the smallest and fastest variant. You can use other CLIP-compatible models by exporting them to ONNX with the same input/output layout and pointing to them via `.modelId()` or `.modelSource()`.

Possible variants (not yet tested with inference4j):

| Model | Source | Embedding dim | Notes |
|-------|--------|---------------|-------|
| `openai/clip-vit-base-patch16` | OpenAI | 512 | 16×16 patches — better quality, ~2× slower |
| `openai/clip-vit-large-patch14` | OpenAI | 768 | Best quality from OpenAI, significantly larger |
| `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | OpenCLIP | 512 | Trained on LAION-2B, often outperforms OpenAI's original |
| `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | OpenCLIP | 768 | Large variant trained on LAION-2B |
| `google/siglip-base-patch16-224` | Google | 768 | SigLIP — improved training objective, strong zero-shot performance |

!!! note
    Models with different embedding dimensions (e.g., 768 instead of 512) will work — the wrappers don't assume a fixed size. However, you must use the same model for both image and text encoding since embeddings are only comparable within the same model's vector space.

## Model details

| Property | Value |
|----------|-------|
| Architecture | ViT-B/32 (vision) + Transformer (text) |
| Embedding dimensions | 512 |
| Image input | 224×224 RGB, CLIP-normalized |
| Text input | BPE tokenized, max 77 tokens |
| Default model | [`inference4j/clip-vit-base-patch32`](https://huggingface.co/inference4j/clip-vit-base-patch32) |
| Model size | ~340 MB (vision) + ~255 MB (text) |
