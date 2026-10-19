# Zero-Shot Named Entity Recognition

Extract entities of types you name at call time, such as "invoice id", "medication" or "email", with no training data
and no fine-tuning. inference4j runs [GLiNER](https://github.com/urchade/GLiNER), including a model tuned for personal
data (PII).

## Quick example

```java
try (var ner = GlinerEntityRecognizer.builder().build()) {
    List<NamedEntity> entities = ner.recognize(
        "Alice Johnson paid invoice INV-2024-0042 to Acme Corp on 3 March.",
        List.of("person", "organization", "invoice id", "date"));
    // Alice Johnson (person), INV-2024-0042 (invoice id), Acme Corp (organization), 3 March (date)
}
```

## Full example

```java
import io.github.inference4j.nlp.GlinerEntityRecognizer;
import io.github.inference4j.nlp.NamedEntity;
import io.github.inference4j.nlp.ZeroShotEntityRecognizer;
import java.util.List;

public class PiiFinder {
    public static void main(String[] args) {
        String message = "Hi, I'm Alice Johnson. Reach me at alice.johnson@example.com or +1 415 555 0132. "
                + "My card is 4111 1111 1111 1111 and I live at 42 Elm Street, Springfield.";

        try (ZeroShotEntityRecognizer pii = GlinerEntityRecognizer.pii().threshold(0.3f).build()) {
            for (NamedEntity e : pii.recognize(message, List.of("person", "email", "phone number",
                    "credit card number", "address"))) {
                System.out.printf("%-20s %-28s %.2f%n", e.label(), e.text(), e.score());
            }
        }
    }
}
```

```
person               Alice Johnson                0.62
email                alice.johnson@example.com    0.98
phone number         +1 415 555 0132              0.99
credit card number   4111 1111 1111 1111          0.96
address              42 Elm Street, Springfield   0.93
```

## Default labels

`recognize(String)`, without labels, looks for the preset's default labels, so a GLiNER recognizer also works as a
`NamedEntityRecognizer`, in place of `BertNerRecognizer`. Set your own with `.labels(List)`.

| Preset | Default labels |
|--------|----------------|
| `small()` (default) | person, organization, location |
| `pii()` | person, email, phone number, address, date of birth, credit card number, bank account number, social security number, passport number, ip address |

```java
try (NamedEntityRecognizer ner = GlinerEntityRecognizer.builder()
        .labels(List.of("person", "company", "product"))
        .build()) {
    ner.recognize("Marie Curie bought an Acme Rocket from Acme Corp.");
    // Marie Curie (person), Acme Rocket (product), Acme Corp (company)
}
```

## How it works

GLiNER reads the entity types and the text together, as `<<ENT>> person <<ENT>> invoice id <<SEP>> Alice Johnson paid
…`, and scores every span of up to 12 words against every type, all in one model run. Spans above the threshold are
taken best first, and a span is dropped if it overlaps one already taken, so entities never overlap. Results come back
in text order, each with the probability of its whole span.

Labels are read as plain language: "phone number" works better than "PHONE" or "tel". Results depend on the wording,
so try a few on your own texts. Every type competes for the same span: offer "organization" alongside "person", or the
model may file a company under "person".

## Models

| Preset | Model | Size | Time per sentence* | License |
|--------|-------|------|--------------------|---------|
| `small()` (default) | `inference4j/gliner-small-v2.1`, from [urchade/gliner_small-v2.1](https://huggingface.co/urchade/gliner_small-v2.1) | ~610 MB | 9.5 ms | Apache 2.0 |
| `pii()` | `inference4j/gliner-pii-base-v1.0`, from [knowledgator/gliner-pii-base-v1.0](https://huggingface.co/knowledgator/gliner-pii-base-v1.0) | ~665 MB | 10.3 ms | Apache 2.0 |

\*Laptop CPU, four ONNX Runtime threads, short news sentences.

The PII model is tuned for personal, health and payment data. Its authors suggest a threshold of 0.3.

### Accuracy on CoNLL-2003

On the first 1,000 sentences of the [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) test set (1,641
person, organization and location entities; exact span and type must match, MISC not scored), labels
"person", "organization" and "location", threshold 0.5:

| Model | F1 | Person | Organization | Location |
|-------|----|--------|--------------|----------|
| `small()` | 0.683 | 0.892 | 0.452 | 0.626 |
| `pii()` | 0.681 | 0.870 | 0.503 | 0.619 |
| `BertNerRecognizer` (distilbert-NER, trained on CoNLL-2003) | 0.859 | 0.871 | 0.866 | 0.832 |

GLiNER finds people as well as a model trained on this data does. Most of its organization and location errors follow
CoNLL's conventions rather than plain reading: in sports reports, CoNLL labels "Japan" or "England" a location, where
GLiNER reads a team (organization). GLiNER also tags pronouns such as "he" as a person. A threshold of 0.3 lowered F1
on this data for both models. **For the fixed person, organization and location types, a trained model such as
`BertNerRecognizer` is more accurate; use GLiNER for types no trained model covers.**

!!! warning "PII detection is not a guarantee"
    Detection is probabilistic: no model finds every name, number or address. Use GLiNER to reduce the personal data
    in text, not as the only safeguard where a missed entity matters, and check it on your own data.

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/gliner-small-v2.1` | HuggingFace model ID; presets `small()`, `pii()` |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.labels(List<String>)` | `List<String>` | the preset's labels | Entity types `recognize(String)` looks for |
| `.threshold(float)` | `float` | `0.5` | Spans scoring at or below this are ignored; in [0, 1) |
| `.multiLabel(boolean)` | `boolean` | `false` | Let the exact same span carry several labels |
| `.truncation(TruncationPolicy)` | `TruncationPolicy` | `TRUNCATE` | What to do with text over the model's word limit |
| `.stride(int)` | `int` | off | Read long text in full, in windows sharing this many words |

## Long text

GLiNER counts words, not tokens: `small()` reads at most 384 words, `pii()` 2,048. By default, longer text is cut and
a warning is logged once; `truncation(TruncationPolicy.FAIL)` throws `InputTooLongException` instead. With
`.stride(int)`, the whole text is read in overlapping windows of words, and each entity is taken from the window where
it starts away from the edges.

```java
try (var ner = GlinerEntityRecognizer.builder().stride(64).build()) {
    ner.recognize(longContract, List.of("person", "organization", "date", "amount"));
}
```

## Result type

`NamedEntity` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `text()` | `String` | The entity text, as it appears in the input |
| `label()` | `String` | One of the labels passed in |
| `start()` | `int` | Character offset start in the original string |
| `end()` | `int` | Character offset end (exclusive) in the original string |
| `score()` | `float` | Probability of the whole span (0.0 to 1.0) |

## Accuracy against the gliner package

On the same ONNX models, inference4j returns the same entities as the `gliner` Python package (0.2.29), with scores
within 1 × 10⁻⁶, on the test texts, at thresholds 0.5 and 0.3, including text truncated at the word limit.
