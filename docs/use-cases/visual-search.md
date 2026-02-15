# Visual Search

Encode images and text into a shared vector space for image-text similarity, zero-shot classification, and visual search — powered by [CLIP](https://arxiv.org/abs/2103.00020).

## Quick start

### Image-text similarity

```java
try (ClipImageEncoder imageEncoder = ClipImageEncoder.builder().build();
     ClipTextEncoder textEncoder = ClipTextEncoder.builder().build()) {

    float[] imageEmb = imageEncoder.encode(ImageIO.read(Path.of("photo.jpg").toFile()));
    float[] textEmb = textEncoder.encode("a photo of a sunset");

    float similarity = dot(imageEmb, textEmb);
    System.out.println("Similarity: " + similarity);
}
```

### Zero-shot classification

Classify images using arbitrary text labels — no training required:

```java
try (ClipImageEncoder imageEncoder = ClipImageEncoder.builder().build();
     ClipTextEncoder textEncoder = ClipTextEncoder.builder().build()) {

    float[] imageEmb = imageEncoder.encode(photo);

    String[] labels = {"cat", "dog", "bird", "car", "airplane"};
    String bestLabel = null;
    float bestScore = Float.NEGATIVE_INFINITY;

    for (String label : labels) {
        float score = dot(imageEmb, textEncoder.encode("a photo of a " + label));
        if (score > bestScore) {
            bestScore = score;
            bestLabel = label;
        }
    }
    System.out.println("Predicted: " + bestLabel);
}
```

### Image search

Find the best-matching images for a text query:

```java
// Index: encode all images once
List<float[]> imageEmbeddings = imageEncoder.encodeBatch(images);

// Query: encode the search text
float[] queryEmb = textEncoder.encode("a red sports car");

// Rank by similarity
int bestIdx = 0;
float bestScore = Float.NEGATIVE_INFINITY;
for (int i = 0; i < imageEmbeddings.size(); i++) {
    float score = dot(queryEmb, imageEmbeddings.get(i));
    if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
    }
}
```

## How it works

CLIP uses two separate encoders — one for images, one for text — trained so that matching image-text pairs produce similar embeddings. Both encoders output 512-dimensional L2-normalized vectors. Cosine similarity (equivalent to dot product for normalized vectors) measures how well an image matches a text description.

```
Image  → ClipImageEncoder → [512-dim vector] ─┐
                                                ├─ dot product → similarity score
Text   → ClipTextEncoder  → [512-dim vector] ─┘
```

## Builder options

### ClipImageEncoder

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelId(String)` | `String` | `inference4j/clip-vit-base-patch32` | HuggingFace model ID |
| `modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Where to load the model from |
| `sessionOptions(SessionConfigurer)` | `SessionConfigurer` | Default (CPU) | ONNX Runtime session options |
| `preprocessor(Preprocessor)` | `Preprocessor` | CLIP pipeline (224×224, CLIP normalization) | Custom image preprocessor |

### ClipTextEncoder

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelId(String)` | `String` | `inference4j/clip-vit-base-patch32` | HuggingFace model ID |
| `modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Where to load the model from |
| `sessionOptions(SessionConfigurer)` | `SessionConfigurer` | Default (CPU) | ONNX Runtime session options |
| `tokenizer(Tokenizer)` | `Tokenizer` | Auto-loaded BPE from model directory | Custom tokenizer |

## Helper: dot product

```java
static float dot(float[] a, float[] b) {
    float sum = 0f;
    for (int i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

Since both encoders produce L2-normalized vectors, the dot product equals cosine similarity.

## Model details

| Property | Value |
|----------|-------|
| Architecture | ViT-B/32 (vision) + Transformer (text) |
| Embedding dimensions | 512 |
| Image input | 224×224 RGB, CLIP-normalized |
| Text input | BPE tokenized, max 77 tokens |
| Default model | [`inference4j/clip-vit-base-patch32`](https://huggingface.co/inference4j/clip-vit-base-patch32) |
| Model size | ~340 MB (vision) + ~255 MB (text) |
