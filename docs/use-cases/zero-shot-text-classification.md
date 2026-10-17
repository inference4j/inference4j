# Zero-Shot Text Classification

Classify text into labels you choose at call time, with no training data and no fine-tuning: support tickets into
queues, articles into topics, feedback into themes. Change the labels and the classifier follows.

## Quick example

```java
try (var classifier = DebertaZeroShotClassifier.builder().build()) {
    List<TextClassification> result = classifier.classify(
        "My invoice shows the same charge twice.",
        List.of("billing", "outage", "feature request", "account access"));
    // billing 0.97, account access 0.02, outage 0.01, feature request 0.00
}
```

## Full example

```java
import io.github.inference4j.nlp.DebertaZeroShotClassifier;
import io.github.inference4j.nlp.TextClassification;
import io.github.inference4j.nlp.ZeroShotTextClassifier;
import java.util.List;

public class TicketRouter {
    public static void main(String[] args) {
        List<String> queues = List.of("billing", "outage", "feature request", "account access");

        try (ZeroShotTextClassifier classifier = DebertaZeroShotClassifier.builder()
                .hypothesisTemplate("The customer is asking about {}.")
                .build()) {
            for (String ticket : List.of(
                    "The service has been unavailable since this morning and nothing loads.",
                    "Please add a dark mode to the app.")) {
                TextClassification top = classifier.classify(ticket, queues).get(0);
                System.out.printf("%-75s -> %s (%.0f%%)%n", ticket, top.label(), top.confidence() * 100);
            }
        }
    }
}
```

## How it works

The classifier is a natural language inference (NLI) model: given a text and a statement, it judges whether the text
**entails** the statement. Each candidate label becomes a statement through the hypothesis template — "This text is
about billing." — and the label's score is how strongly the text entails it. This is how HuggingFace's
`zero-shot-classification` pipeline works, and inference4j's scores match it.

All of a text's labels are scored in one batch, so one call costs one model run per text, whatever the number of
labels (up to the batch's memory).

## Single-label and multi-label

| Mode | Scores | Use it when |
|------|--------|-------------|
| `multiLabel(false)` (default) | A softmax over the labels: they compete and sum to 1 | Exactly one label applies (routing, topic) |
| `multiLabel(true)` | Each label judged on its own (entailment against contradiction), 0 to 1 | Several labels, or none, may apply (tagging) |

```java
try (var tagger = DebertaZeroShotClassifier.builder().multiLabel(true).build()) {
    tagger.classify("I can't log in after resetting my password, and I was charged for a plan I cancelled.",
        List.of("billing", "account access", "outage"));
    // account access 0.97, billing 0.27, outage 0.03
}
```

With a single candidate label, the score is always computed the multi-label way: a label cannot compete with itself.

## Writing labels and templates

- **Labels are read as plain language.** "feature request" works better than "FR" or "feat_req".
- **The template matters.** The default, `"This text is about {}."`, suits topics. A template written for your texts,
  such as `"The customer is asking about {}."` or `"The sentiment of this review is {}."`, often scores better.
  `{}` stands for the label.
- **Validate on your own data.** Zero-shot accuracy varies a lot between tasks. Try a few templates and label
  wordings on a sample of labeled texts before relying on it.

## Models

| Preset | Model | Size | AG News accuracy* | Time per text* |
|--------|-------|------|-------------------|----------------|
| `base()` (default) | `inference4j/deberta-v3-base-zeroshot-v2.0-c` | ~740 MB | 64.0% | 47 ms |
| `large()` | `inference4j/deberta-v3-large-zeroshot-v2.0-c` | ~1.7 GB | 78.7% | 150 ms |

\*400 texts from the [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) test split, four topics (World, Sports,
Business, Science and Technology), default template, single-label; laptop CPU, four ONNX Runtime threads. On the
author's broader benchmark of 28 classification tasks, the models reach a mean F1 of 0.619 (base) and 0.676 (large),
against 0.497 for `facebook/bart-large-mnli`.

Both are [Moritz Laurer's zero-shot v2.0 "-c" models](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c):
MIT-licensed and trained **only on commercially-friendly data** (synthetic data plus the MNLI and FEVER-NLI datasets).

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/deberta-v3-base-zeroshot-v2.0-c` | HuggingFace model ID; presets `base()`, `large()` |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.hypothesisTemplate(String)` | `String` | `"This text is about {}."` | How a label becomes a hypothesis; `{}` is the label |
| `.multiLabel(boolean)` | `boolean` | `false` | Score labels independently instead of letting them compete |
| `.maxLength(int)` | `int` | `512` | Maximum tokens per model run, text, hypothesis and special tokens included |
| `.stride(int)` | `int` | off | Score long text in full, in windows sharing this many tokens |
| `.windowAggregation(WindowAggregation)` | `WindowAggregation` | `mean()` single-label, `max()` multi-label | How window scores combine. Requires `.stride(int)` |

## Long text

By default, text that does not fit in `maxLength` tokens together with a hypothesis is cut at its end; the hypothesis
is always kept whole, as in HuggingFace. With `.stride(int)`, the whole text is scored in overlapping windows and the
window scores combined: by default `mean()` for single-label (an overall verdict) and `max()` for multi-label (a label
that applies anywhere in the text applies to the text).

```java
try (var classifier = DebertaZeroShotClassifier.builder().stride(128).build()) {
    classifier.classify(longReport, List.of("finance", "legal", "engineering"));
}
```

## Result type

`TextClassification` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `label()` | `String` | The candidate label |
| `index()` | `int` | The label's position in the list passed in |
| `confidence()` | `float` | The label's score (0.0 to 1.0) |

Results are sorted by confidence, highest first.

## Accuracy against HuggingFace

On the same ONNX model, inference4j's scores equal those of HuggingFace's tokenizer and post-processing to within
4 × 10⁻⁷. Compared with the `zero-shot-classification` pipeline running the original PyTorch weights, scores differ by
up to 3 × 10⁻⁴ (more on texts near 512 tokens), the ordinary difference between ONNX and PyTorch arithmetic; the top
label agreed on all 400 AG News texts and all test cases, and accuracy was identical.
