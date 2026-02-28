# Machine Translation

Translate text between languages using MarianMT (fixed language pairs) or Flan-T5 (flexible, any-to-any).

## Quick example

=== "MarianMT (fixed pair)"

    ```java
    try (var translator = MarianTranslator.builder()
            .modelId("inference4j/opus-mt-en-fr")
            .build()) {
        String french = translator.translate("The weather is beautiful today.");
        System.out.println(french); // Le temps est beau aujourd'hui.
    }
    ```

=== "Flan-T5 (flexible)"

    ```java
    try (var translator = FlanT5TextGenerator.flanT5Base().build()) {
        String french = translator.translate("The weather is beautiful today.",
                Language.EN, Language.FR);
        System.out.println(french);
    }
    ```

## Full example

```java
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.nlp.MarianTranslator;

public class Translation {
    public static void main(String[] args) {
        try (var translator = MarianTranslator.builder()
                .modelId("inference4j/opus-mt-en-de")
                .maxNewTokens(200)
                .build()) {

            GenerationResult result = translator.translate(
                    "Machine learning is transforming how we build software.",
                    token -> System.out.print(token));

            System.out.println();
            System.out.printf("%d tokens in %,d ms%n",
                    result.generatedTokens(), result.duration().toMillis());
        }
    }
}
```

## Flexible translation with Flan-T5

`FlanT5TextGenerator` implements the `Translator` interface and can translate between any pair of languages using a single model:

```java
import io.github.inference4j.nlp.FlanT5TextGenerator;
import io.github.inference4j.nlp.Language;

try (var translator = FlanT5TextGenerator.flanT5Base()
        .maxNewTokens(200)
        .build()) {

    // English to French
    String french = translator.translate("Hello, how are you?",
            Language.EN, Language.FR);

    // English to German
    String german = translator.translate("Hello, how are you?",
            Language.EN, Language.DE);

    // French to Spanish
    String spanish = translator.translate("Bonjour, comment allez-vous?",
            Language.FR, Language.ES);
}
```

## Supported languages

The `Language` enum provides 24 language constants:

| Category | Languages |
|----------|-----------|
| Western European | `EN`, `FR`, `DE`, `ES`, `PT`, `PT_BR`, `IT`, `NL`, `CA` |
| Northern European | `SV`, `DA`, `NO`, `FI` |
| Eastern European | `PL`, `CS`, `HR`, `RO` |
| Other | `TR`, `JA`, `KO`, `AR`, `ZH_CN`, `ZH_TW`, `HI` |

Each constant provides `displayName()` (e.g., `"Brazilian Portuguese"`) and `isoCode()` (e.g., `"pt-br"`).

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | — (required for MarianMT) | HuggingFace model ID (e.g., `inference4j/opus-mt-en-fr`) |
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
| `text()` | `String` | The translated text |
| `promptTokens()` | `int` | Number of tokens in the input |
| `generatedTokens()` | `int` | Number of tokens generated |
| `duration()` | `Duration` | Wall-clock generation time |

The convenience method `translate(text)` returns the translation as a plain `String`.

## Tips

- **MarianMT** models are specialized for a single language pair (e.g., `opus-mt-en-fr` for English→French). They produce higher quality translations for their specific pair but require a separate model per direction.
- **Flan-T5** handles any language pair with a single model, making it more flexible but generally lower quality than a dedicated pair-specific model.
- MarianMT models are available on HuggingFace under the `Helsinki-NLP` organization. Export them to ONNX and host under your own org, or use pre-exported models from `inference4j`.
- For bidirectional translation, you need two MarianMT models (e.g., `opus-mt-en-fr` and `opus-mt-fr-en`) — or use Flan-T5 which handles both directions.
- Use greedy decoding (default `temperature=0`) for translation — sampling adds noise without improving quality.
