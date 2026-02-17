# Tokenizers

Transformer models don't work with raw text — they expect sequences of integer token IDs mapped from a fixed vocabulary. A **tokenizer** bridges this gap: it splits text into tokens, maps each token to its vocabulary index, and produces the metadata tensors (`attention_mask`, `token_type_ids`) that the model expects as input.

inference4j handles tokenization automatically. When you call `classifier.classify("some text")`, the wrapper tokenizes the input, runs inference, and decodes the output — you never touch a token ID. But if you need to customize tokenization or use tokenizers directly, this guide covers how.

## Built-in tokenizers

inference4j ships two tokenizer implementations, covering the two most common algorithms in production transformer models:

| Tokenizer | Algorithm | Models | Vocabulary format |
|-----------|-----------|--------|-------------------|
| `WordPieceTokenizer` | WordPiece (greedy longest-match subword splitting) | BERT, DistilBERT, MiniLM, SentenceTransformer | `vocab.txt` (one token per line) |
| `BpeTokenizer` | Byte-level BPE (iterative pair merging) | CLIP, GPT-2 | `vocab.json` + `merges.txt` |

### WordPiece

WordPiece breaks unknown words into known subword units using a `##` continuation prefix. For example, `"unbelievable"` becomes `["un", "##believ", "##able"]` — preserving meaning even for out-of-vocabulary words.

The encoding pipeline:

1. Lowercase and split on whitespace/punctuation
2. Split each word into subwords via greedy longest-match against the vocabulary
3. Wrap with `[CLS]` and `[SEP]` special tokens
4. Truncate to `maxLength` if needed

```java
Tokenizer tokenizer = WordPieceTokenizer.fromVocabFile(Path.of("vocab.txt"));
EncodedInput encoded = tokenizer.encode("Hello world!", 128);
// encoded.inputIds()      → [101, 7592, 2088, 999, 102]
// encoded.attentionMask() → [1, 1, 1, 1, 1]
// encoded.tokenTypeIds()  → [0, 0, 0, 0, 0]
```

WordPiece also supports **sentence pair encoding** for cross-encoder models (e.g., rerankers):

```java
EncodedInput encoded = tokenizer.encode("What is Java?", "Java is a programming language.", 128);
// Format: [CLS] textA [SEP] textB [SEP]
// tokenTypeIds: 0 for textA tokens, 1 for textB tokens
```

!!! note
    The built-in `WordPieceTokenizer` applies unconditional lowercasing, matching `bert-base-uncased` and `distilbert-base-uncased`. It is not suitable for cased models.

### Byte-Pair Encoding (BPE)

BPE starts from individual characters and iteratively merges the most frequent pairs into subwords. CLIP's variant adds byte-level encoding (handling any UTF-8 input) and `</w>` end-of-word markers.

The encoding pipeline:

1. Lowercase and normalize whitespace
2. Split via regex into words, contractions, digits, and punctuation
3. Encode each byte via GPT-2's byte-to-unicode table
4. Apply BPE merges according to the priority table
5. Wrap with `<|startoftext|>` and `<|endoftext|>` special tokens
6. Pad to `maxLength` (default: 77 for CLIP)

```java
Tokenizer tokenizer = BpeTokenizer.fromFiles(
        Path.of("vocab.json"), Path.of("merges.txt"));
EncodedInput encoded = tokenizer.encode("a photo of a cat");
// encoded.inputIds()      → [49406, 320, 1125, 539, 320, 2368, 49407, 0, ...]
// encoded.attentionMask() → [1, 1, 1, 1, 1, 1, 1, 0, ...]
```

## Default behavior

You don't need to configure tokenizers for standard use. Every NLP wrapper auto-loads the correct tokenizer from the model directory during `.build()`:

```java
// WordPiece loaded automatically from vocab.txt
try (var classifier = DistilBertTextClassifier.builder()
        .modelId("inference4j/distilbert-base-uncased-finetuned-sst-2-english")
        .build()) {
    classifier.classify("This movie was fantastic!");
}

// BPE loaded automatically from vocab.json + merges.txt
try (var classifier = ClipClassifier.builder().build()) {
    classifier.classify(Path.of("photo.jpg"), List.of("cat", "dog", "bird"));
}
```

The wrapper knows which tokenizer algorithm its model expects and which vocabulary files to look for.

## Supplying a custom tokenizer

All NLP builders expose a `.tokenizer()` method that lets you override the default:

```java
Tokenizer myTokenizer = WordPieceTokenizer.fromVocabFile(Path.of("/path/to/my/vocab.txt"));

try (var embedder = SentenceTransformerEmbedder.builder()
        .modelId("my-custom-model")
        .tokenizer(myTokenizer)
        .build()) {
    float[] embedding = embedder.encode("Hello, world!");
}
```

When you provide a tokenizer, the wrapper skips auto-loading and uses yours directly.

### When to use a custom tokenizer

**Shared instances** — if you're running multiple wrappers against the same vocabulary, share a single tokenizer instance to avoid loading the vocabulary file multiple times:

```java
Tokenizer shared = WordPieceTokenizer.fromVocabFile(Path.of("vocab.txt"));

try (var embedder = SentenceTransformerEmbedder.builder()
            .modelId("my-model").tokenizer(shared).build();
     var classifier = DistilBertTextClassifier.builder()
            .modelId("my-model").tokenizer(shared).build()) {
    // Both use the same tokenizer instance
}
```

**Custom vocabulary** — if you've fine-tuned a model with a modified vocabulary, point the tokenizer at your custom `vocab.txt`:

```java
Tokenizer tokenizer = WordPieceTokenizer.fromVocabFile(Path.of("my-custom-vocab.txt"));
try (var classifier = DistilBertTextClassifier.builder()
        .tokenizer(tokenizer)
        .modelSource(LocalModelSource.of(Path.of("my-finetuned-model")))
        .build()) {
    classifier.classify("custom domain text");
}
```

**Testing** — supply a mock or stub tokenizer in unit tests to isolate inference logic from tokenization:

```java
Tokenizer stub = text -> new EncodedInput(
    new long[]{101, 7592, 102},
    new long[]{1, 1, 1},
    new long[]{0, 0, 0}
);
```

## The `EncodedInput` record

Both tokenizers return an `EncodedInput` containing the three standard tensors that transformer models expect:

```java
public record EncodedInput(
    long[] inputIds,       // token IDs from the vocabulary
    long[] attentionMask,  // 1 for real tokens, 0 for padding
    long[] tokenTypeIds    // segment IDs (0 for first sentence, 1 for second)
) {}
```

| Field | Purpose | Example |
|-------|---------|---------|
| `inputIds` | Maps each token to its vocabulary index | `[101, 7592, 2088, 102]` |
| `attentionMask` | Tells the model which positions are real tokens vs padding | `[1, 1, 1, 1]` |
| `tokenTypeIds` | Distinguishes sentence A from sentence B in pair tasks | `[0, 0, 0, 0]` |

## Tips

- **You almost never need to touch tokenizers.** The default auto-loading handles standard HuggingFace models out of the box.
- **Tokenizer and model must match.** A tokenizer trained on one vocabulary will produce wrong token IDs for a model trained on a different vocabulary. Always use the tokenizer that shipped with your model.
- **`maxLength` matters.** Most BERT models use 512 tokens max. CLIP uses 77. Exceeding the model's trained length produces undefined results. The wrappers set this automatically.
- **WordPiece does not pad, BPE does.** WordPiece returns only the actual tokens (variable length). BPE pads to `maxLength` with zeros. Both behaviors match what their respective model families expect.
