# PII redaction

`Inference4jPiiRedactor` replaces personal data in LangChain4j documents and text segments with placeholders such as
`[EMAIL]`, in-process, before the text is embedded, stored, or sent to a hosted model.

```text
Before:  Alice Johnson (alice.johnson@example.com, +1 415 555 0134) reported a billing error.
After:   [PERSON] ([EMAIL], [PHONE_NUMBER]) reported a billing error.
```

The example shows the placeholder format. Detected entities depend on the model and its threshold.

## Quick example

```java
try (var redactor = Inference4jPiiRedactor.builder()
        .recognizer(GlinerEntityRecognizer.pii().threshold(0.3f).stride(64).build())
        .build()) {

    EmbeddingStoreIngestor.builder()
            .documentTransformer(redactor)
            .documentSplitter(DocumentSplitters.recursive(300, 30))
            .embeddingModel(embeddingModel)
            .embeddingStore(store)
            .build()
            .ingest(documents);
}
```

Redaction runs before the splitter and the embedding model, so the embedding provider, the embedding store and
the chat model that receives retrieved segments get the redacted text.

!!! warning "Redaction is not a guarantee"
    Detection is probabilistic: no model finds every entity. Redaction reduces the personal data in text; it does
    not guarantee that none remains. Two recognizer settings affect how much is found:

    - **`threshold(0.3f)`**: the value the PII model's authors suggest. Lower finds more entities, with some false
      positives. The default is `0.5`.
    - **`stride(64)`**: the PII model reads at most 2,048 words. Without `stride`, longer text is cut and **the rest is
      not redacted**. With it, long text is read in full in overlapping windows.

## Documents or segments

The redactor is a `DocumentTransformer`. For the segment slot of the ingestor, use its segment view:

```java
EmbeddingStoreIngestor.builder()
        .documentSplitter(DocumentSplitters.recursive(300, 30))
        .textSegmentTransformer(redactor.asTextSegmentTransformer())
        .embeddingModel(embeddingModel)
        .embeddingStore(store)
        .build();
```

| Ingestor slot | Use when |
|-------|-------------------|
| `documentTransformer(redactor)` | The recognizer reads whole documents, so the splitter cannot cut an entity in two |
| `textSegmentTransformer(redactor.asTextSegmentTransformer())` | Segments are produced elsewhere, or redaction counts are needed per segment |

The view shares the redactor's recognizer and stops working when the redactor is closed. Use one of the two slots.

`redactor.redact(String)` redacts a plain string.

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.recognizer(NamedEntityRecognizer)` | `NamedEntityRecognizer` | — (required) | The model that finds personal data. Typically `GlinerEntityRecognizer.pii().threshold(0.3f).stride(64).build()` |
| `.labels(Collection<String>)` | `Collection<String>` | every entity found | Redact only these labels (case-insensitive), e.g. `List.of("email", "phone number")` |
| `.minScore(float)` | `float` | unset | Redact only entities scoring at least this, on top of the recognizer's threshold. To find more entities, lower the recognizer's threshold |
| `.placeholder(Function<NamedEntity, String>)` | `Function` | `[LABEL]` in upper snake case | What replaces each span, e.g. `entity -> "<redacted>"` |

## What gets redacted

With `GlinerEntityRecognizer.pii()` the labels are: person, email, phone number, address, date of birth, credit card
number, bank account number, social security number, passport number, ip address. Change them with `.labels(...)` on
the recognizer; GLiNER is [zero-shot](../../use-cases/zero-shot-ner.md) and accepts any entity type by name.

Spans are merged before they are replaced, so one entity gives one placeholder:

- spans that overlap or touch become one span;
- spans with the same label separated only by whitespace become one span ("Alice" and "Johnson" give one `[PERSON]`);
- a merged span takes the label of its highest-scoring part.

## Metadata

Redacted documents and segments keep their metadata and gain these keys:

| Key | Type | Value |
|-----|------|-------|
| `pii_redacted_count` (`Inference4jPiiRedactor.REDACTED_COUNT`) | `int` | Number of placeholders written; `0` when nothing was found |
| `pii_redacted_<label>` (`Inference4jPiiRedactor.REDACTED_PREFIX` + label) | `int` | Count per label found, in snake case: `pii_redacted_email`, `pii_redacted_phone_number` |

```java
Document redacted = redactor.transform(document);
Integer count = redacted.metadata().getInteger(Inference4jPiiRedactor.REDACTED_COUNT);
```

## Model choices

| Recognizer | Size | Notes |
|------------|------|-------|
| `GlinerEntityRecognizer.pii()` | ~665 MB | Tuned for personal, health and payment data |
| `GlinerEntityRecognizer.small()` | ~610 MB | General zero-shot NER with caller-defined labels |
| `BertNerRecognizer.builder()` | ~260 MB | Fixed types: person, organization, location, miscellaneous. Does not detect emails, phone numbers or card numbers |

Any `NamedEntityRecognizer` works. Measured accuracy is under
[Zero-shot NER](../../use-cases/zero-shot-ner.md).

## Chat messages

The redactor implements `DocumentTransformer` and provides a `TextSegmentTransformer` view. It is not an
input or output guardrail and does not rewrite chat messages. Use `redact(String)` to redact text elsewhere in
the application, for example before logging a conversation.

Tools receive the values in the user's message. Redacting a message removes values that a tool call may need,
such as a name or a booking number.

## Notes

- **Redaction is one-way.** Placeholders are not mapped back to the original values.
- **Retrieval matches redacted text.** A query for "Alice Johnson" does not match a segment that contains
  `[PERSON]`. Use `.labels(...)` to limit what is redacted.
- **Cost.** One recognizer run per document (more with `stride` on long documents), on CPU by default.
- **Lifecycle.** `close()` closes the recognizer. See
  [Lifecycle](index.md#lifecycle).
