# CLIP Encoders

Low-level access to CLIP's vision and text encoders for image-text similarity, visual search, and custom retrieval pipelines.

For zero-shot image classification, use [`ClipClassifier`](../use-cases/visual-search.md) instead — it provides a higher-level `ImageClassifier` API on top of these encoders.

## ClipImageEncoder

Maps images to 512-dimensional L2-normalized embeddings.

```java
try (ClipImageEncoder encoder = ClipImageEncoder.builder().build()) {
    float[] embedding = encoder.encode(ImageIO.read(Path.of("photo.jpg").toFile()));
    // 512-dim L2-normalized vector
}
```

### Batch encoding

```java
List<float[]> embeddings = encoder.encodeBatch(images);
```

### Builder options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelId(String)` | `String` | `inference4j/clip-vit-base-patch32` | HuggingFace model ID |
| `modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Where to load the model from |
| `sessionOptions(SessionConfigurer)` | `SessionConfigurer` | Default (CPU) | ONNX Runtime session options |
| `preprocessor(Preprocessor)` | `Preprocessor` | CLIP pipeline (224×224, CLIP normalization) | Custom image preprocessor |

### Preprocessing

- Resize to 224×224, center crop
- CLIP normalization: mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`
- NCHW layout: `[1, 3, 224, 224]`

## ClipTextEncoder

Maps text to 512-dimensional L2-normalized embeddings in the same vector space as `ClipImageEncoder`.

```java
try (ClipTextEncoder encoder = ClipTextEncoder.builder().build()) {
    float[] embedding = encoder.encode("a photo of a cat");
    // 512-dim L2-normalized vector
}
```

### Builder options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelId(String)` | `String` | `inference4j/clip-vit-base-patch32` | HuggingFace model ID |
| `modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Where to load the model from |
| `sessionOptions(SessionConfigurer)` | `SessionConfigurer` | Default (CPU) | ONNX Runtime session options |
| `tokenizer(Tokenizer)` | `Tokenizer` | Auto-loaded BPE from model directory | Custom tokenizer |

### Tokenization

Uses byte-level BPE tokenization (`BpeTokenizer`) with CLIP's vocabulary. The tokenizer is automatically loaded from `vocab.json` and `merges.txt` in the model directory. Sequences are wrapped with BOS/EOS tokens and padded to 77 tokens.

## Image-text similarity

```java
try (ClipImageEncoder imageEncoder = ClipImageEncoder.builder().build();
     ClipTextEncoder textEncoder = ClipTextEncoder.builder().build()) {

    float[] imageEmb = imageEncoder.encode(ImageIO.read(Path.of("photo.jpg").toFile()));
    float[] textEmb = textEncoder.encode("a photo of a sunset");

    float similarity = dot(imageEmb, textEmb);
    System.out.println("Similarity: " + similarity);
}
```

## Image search

Index a collection of images, then query with text:

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

## Dot product helper

Since both encoders produce L2-normalized vectors, the dot product equals cosine similarity:

```java
static float dot(float[] a, float[] b) {
    float sum = 0f;
    for (int i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
```
