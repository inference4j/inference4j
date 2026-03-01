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

## Using your own MarianMT model

The pre-exported models under `inference4j/opus-mt-*` work out of the box. If you want to use a different MarianMT language pair (e.g., `Helsinki-NLP/opus-mt-en-ja`), you'll need to export it yourself.

`MarianTranslator` expects the model directory to contain:

| File | Description |
|------|-------------|
| `encoder_model.onnx` | Encoder ONNX model |
| `decoder_model.onnx` | Decoder ONNX model |
| `decoder_with_past_model.onnx` | Decoder with KV cache |
| `config.json` | Model configuration |
| `tokenizer.json` | HuggingFace fast tokenizer format |

!!! warning "MarianMT models require tokenizer conversion"

    MarianMT models on HuggingFace ship with SentencePiece files (`source.spm`, `target.spm`) instead of `tokenizer.json`. You must convert the tokenizer during export. Using Hugging Face Optimum and the `tokenizers` library:

    ```python
    from optimum.exporters.onnx import main_export
    from transformers.convert_slow_tokenizer import SentencePieceExtractor
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    model_id = "Helsinki-NLP/opus-mt-en-ja"

    # 1. Export ONNX models
    main_export(
        model_name_or_path=model_id,
        output="my-model/",
        task="text2text-generation-with-past",
    )

    # 2. Convert source.spm to tokenizer.json
    extractor = SentencePieceExtractor("my-model/source.spm")
    vocab, merges = extractor.extract(None)
    tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))
    tokenizer.save("my-model/tokenizer.json")
    ```

    Only standard `opus-mt-*` models are supported. The newer `opus-mt-tc-big-*` variants require target language prefixes (e.g., `>>por<<`) which `MarianTranslator` does not handle.

## Tips

- **MarianMT** models are specialized for a single language pair (e.g., `opus-mt-en-fr` for English→French). They produce higher quality translations for their specific pair but require a separate model per direction.
- **Flan-T5** handles any language pair with a single model, making it more flexible but generally lower quality than a dedicated pair-specific model.
- For bidirectional translation, you need two MarianMT models (e.g., `opus-mt-en-fr` and `opus-mt-fr-en`) — or use Flan-T5 which handles both directions.
- Use greedy decoding (default `temperature=0`) for translation — sampling adds noise without improving quality.
