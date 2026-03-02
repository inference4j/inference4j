# Named Entity Recognition

Extract named entities (persons, organizations, locations, miscellaneous) from text using a fine-tuned BERT model.

## Quick example

```java
try (var ner = BertNerRecognizer.builder().build()) {
    List<NamedEntity> entities = ner.recognize("John works at Google in London.");
    // [NamedEntity[text=John, label=PER], NamedEntity[text=Google, label=ORG],
    //  NamedEntity[text=London, label=LOC]]
}
```

## Full example

```java
import io.github.inference4j.nlp.BertNerRecognizer;
import io.github.inference4j.nlp.NamedEntity;
import java.util.List;

public class NerExample {
    public static void main(String[] args) {
        try (var ner = BertNerRecognizer.builder().build()) {
            String text = "Marie Curie worked at the University of Paris in France.";
            List<NamedEntity> entities = ner.recognize(text);

            for (NamedEntity entity : entities) {
                System.out.printf("%-20s → %s (%.2f%%)%n",
                    entity.text(), entity.label(), entity.score() * 100);
            }
            // Marie Curie          → PER (99.12%)
            // University of Paris  → ORG (98.45%)
            // France               → LOC (97.89%)
        }
    }
}
```

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/distilbert-NER` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.tokenizer(Tokenizer)` | `Tokenizer` | auto-loaded `WordPieceTokenizer` (cased) | Custom tokenizer |
| `.config(ModelConfig)` | `ModelConfig` | auto-loaded from `config.json` | Model config with IOB2 labels |
| `.maxLength(int)` | `int` | `512` | Maximum token sequence length |

## Result type

`NamedEntity` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `text()` | `String` | The entity span text (e.g., "London") |
| `label()` | `String` | Entity type: `PER`, `ORG`, `LOC`, or `MISC` |
| `start()` | `int` | Character offset start in the original string |
| `end()` | `int` | Character offset end (exclusive) in the original string |
| `score()` | `float` | Mean confidence of the constituent tokens (0.0 to 1.0) |

## Entity types

The default model uses CoNLL-2003 IOB2 labels:

| Label | Description | Examples |
|-------|-------------|----------|
| `PER` | Person | John, Marie Curie, Leonardo da Vinci |
| `ORG` | Organization | Google, United Nations, NASA |
| `LOC` | Location | London, New York, Pacific Ocean |
| `MISC` | Miscellaneous | English, FIFA World Cup, Nobel Prize |

## Available models

| Model | Wrapper | Size | F1 | License |
|-------|---------|------|-----|---------|
| `inference4j/distilbert-NER` | `BertNerRecognizer` | ~260 MB | 92.17 | Apache 2.0 |
| `inference4j/bert-base-NER` | `BertNerRecognizer` | ~431 MB | 91.3 | MIT |

```java
// Use the larger BERT model for slightly different accuracy characteristics
try (var ner = BertNerRecognizer.builder()
        .modelId("inference4j/bert-base-NER")
        .build()) {
    ner.recognize("...");
}
```

## How it works

1. Text is tokenized using a **cased** WordPiece tokenizer (case matters for NER: "Apple" vs "apple")
2. Subword tokens that belong to the same word share a word ID
3. The model predicts an IOB2 label for each token
4. First-subtoken strategy: only the first subtoken's prediction is used for each word
5. `B-*` and `I-*` spans are aggregated into `NamedEntity` objects with character offsets

## Tips

- The default model is **cased** — "Apple" (ORG) and "apple" (fruit) are different tokens. Do not lowercase your input.
- Multi-word entities like "New York" are automatically grouped when the model predicts `B-LOC` followed by `I-LOC`.
- Character offsets (`start()`, `end()`) can be used to highlight entities in the original text.
- For production use, consider the distilbert variant — it's faster with minimal accuracy loss.
