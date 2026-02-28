# Grammar Correction

Fix grammatical errors in text using CoEdIT or Flan-T5 encoder-decoder models.

## Quick example

```java
try (var corrector = CoeditGrammarCorrector.coeditBase().build()) {
    String corrected = corrector.correct("She don't likes swimming.");
    System.out.println(corrected); // She doesn't like swimming.
}
```

## Full example

```java
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.nlp.CoeditGrammarCorrector;

public class GrammarCorrection {
    public static void main(String[] args) {
        try (var corrector = CoeditGrammarCorrector.coeditBase()
                .maxNewTokens(200)
                .build()) {

            String[] sentences = {
                "She don't likes swimming.",
                "Me and him went to the store yesterday.",
                "The informations is very useful for we."
            };

            for (String sentence : sentences) {
                GenerationResult result = corrector.correct(sentence,
                        token -> System.out.print(token));
                System.out.println();
                System.out.printf("  → %d tokens in %,d ms%n",
                        result.generatedTokens(), result.duration().toMillis());
            }
        }
    }
}
```

## Using Flan-T5 as an alternative

`FlanT5TextGenerator` can also correct grammar. It implements the same `GrammarCorrector` interface:

```java
import io.github.inference4j.nlp.FlanT5TextGenerator;
import io.github.inference4j.nlp.GrammarCorrector;

// Both implement GrammarCorrector — swap freely
GrammarCorrector corrector = FlanT5TextGenerator.flanT5Base()
        .maxNewTokens(200)
        .build();
```

## Model presets

### CoeditGrammarCorrector

| Preset | Model | Parameters | Size |
|--------|-------|-----------|------|
| `CoeditGrammarCorrector.coeditBase()` | CoEdIT Base | 250M | ~900 MB |
| `CoeditGrammarCorrector.coeditLarge()` | CoEdIT Large | 780M | ~3 GB |

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
| `.tokenizerProvider(TokenizerProvider)` | `TokenizerProvider` | `SentencePieceBpeTokenizer` | Tokenizer construction strategy |
| `.maxNewTokens(int)` | `int` | `256` | Maximum tokens to generate |
| `.temperature(float)` | `float` | `0.0` | Sampling temperature |
| `.topK(int)` | `int` | `0` (disabled) | Top-K sampling |
| `.topP(float)` | `float` | `0.0` (disabled) | Nucleus sampling |
| `.eosTokenId(int)` | `int` | Auto-detected | End-of-sequence token ID |
| `.addedToken(String)` | `String` | — | Register a special token for atomic encoding |

## Result type

`GenerationResult` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `text()` | `String` | The corrected text |
| `promptTokens()` | `int` | Number of tokens in the input |
| `generatedTokens()` | `int` | Number of tokens generated |
| `duration()` | `Duration` | Wall-clock generation time |

The convenience method `correct(text)` returns the corrected text as a plain `String`.

## Tips

- **CoEdIT** is specifically trained for grammar correction (using the "Fix grammatical errors" instruction internally). It produces more reliable corrections than general-purpose models.
- **Flan-T5** is a general-purpose model that also handles summarization, translation, and SQL generation. Use it when you need multiple tasks from a single model.
- Use greedy decoding (default `temperature=0`) for grammar correction — sampling introduces random variations.
- CoEdIT automatically prepends the instruction prefix `"Fix grammatical errors in this sentence: "` — just pass the raw text to `correct()`.
- For batch correction, reuse the same instance — each call to `correct()` runs an independent generation.
