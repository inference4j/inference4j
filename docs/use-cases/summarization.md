# Text Summarization

Summarize long articles and documents into concise text using BART or Flan-T5 encoder-decoder models.

## Quick example

```java
try (var summarizer = BartSummarizer.distilBartCnn().build()) {
    String summary = summarizer.summarize("Long article text here...");
    System.out.println(summary);
}
```

## Full example

```java
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.nlp.BartSummarizer;

public class Summarization {
    public static void main(String[] args) {
        try (var summarizer = BartSummarizer.distilBartCnn()
                .maxNewTokens(150)
                .build()) {

            String article = """
                The Amazon rainforest, often referred to as the "lungs of the Earth",
                produces about 20% of the world's oxygen. Spanning across nine countries
                in South America, it is the largest tropical rainforest in the world,
                covering approximately 5.5 million square kilometers. The forest is home
                to an estimated 10% of all species on Earth, including over 40,000 plant
                species, 1,300 bird species, and 3,000 types of fish. Deforestation
                remains a critical threat, with an estimated 17% of the forest lost in
                the last 50 years due to logging, agriculture, and urban expansion.
                """;

            GenerationResult result = summarizer.summarize(article, token -> System.out.print(token));
            System.out.println();
            System.out.printf("%d tokens in %,d ms%n",
                    result.generatedTokens(), result.duration().toMillis());
        }
    }
}
```

## Using Flan-T5 as an alternative

`FlanT5TextGenerator` can also summarize text. It uses a different architecture but implements the same `Summarizer` interface:

```java
import io.github.inference4j.nlp.FlanT5TextGenerator;
import io.github.inference4j.nlp.Summarizer;

// Both implement Summarizer — swap freely
Summarizer summarizer = FlanT5TextGenerator.flanT5Base()
        .maxNewTokens(150)
        .build();
```

## Model presets

### BartSummarizer

| Preset | Model | Parameters | Size |
|--------|-------|-----------|------|
| `BartSummarizer.distilBartCnn()` | DistilBART CNN 12-6 | 306M | ~1.2 GB |
| `BartSummarizer.bartLargeCnn()` | BART Large CNN | 406M | ~1.6 GB |

### FlanT5TextGenerator

| Preset | Model | Parameters | Size |
|--------|-------|-----------|------|
| `FlanT5TextGenerator.flanT5Small()` | Flan-T5 Small | 77M | ~300 MB |
| `FlanT5TextGenerator.flanT5Base()` | Flan-T5 Base | 250M | ~900 MB |
| `FlanT5TextGenerator.flanT5Large()` | Flan-T5 Large | 780M | ~3 GB |

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | Preset-dependent | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.tokenizerProvider(TokenizerProvider)` | `TokenizerProvider` | Preset-dependent | Tokenizer construction strategy |
| `.maxNewTokens(int)` | `int` | `256` | Maximum tokens to generate |
| `.temperature(float)` | `float` | `0.0` | Sampling temperature (higher = more random) |
| `.topK(int)` | `int` | `0` (disabled) | Top-K sampling |
| `.topP(float)` | `float` | `0.0` (disabled) | Nucleus sampling |
| `.eosTokenId(int)` | `int` | Auto-detected | End-of-sequence token ID |
| `.addedToken(String)` | `String` | — | Register a special token for atomic encoding |

## Result type

Both `summarize(text, tokenListener)` and `generate(text, tokenListener)` return a `GenerationResult` record:

| Field | Type | Description |
|-------|------|-------------|
| `text()` | `String` | The generated summary |
| `promptTokens()` | `int` | Number of tokens in the input |
| `generatedTokens()` | `int` | Number of tokens generated |
| `duration()` | `Duration` | Wall-clock generation time |

The convenience method `summarize(text)` returns the summary as a plain `String`.

## Tips

- **DistilBART CNN** is purpose-built for summarization and produces the best summaries. Use it when summarization is your only task.
- **Flan-T5** is a general-purpose model that also handles translation, grammar correction, and SQL generation. Use it when you need multiple tasks from a single model.
- Lower `maxNewTokens` for shorter summaries — the model will still produce coherent output.
- Use streaming (`summarize(text, token -> ...)`) for long inputs where generation takes several seconds.
- Reuse instances across calls — each one holds the model and tokenizer in memory.
