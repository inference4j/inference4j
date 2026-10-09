# Punctuation & Casing

Turn raw, unpunctuated text into readable sentences. `PunctCapSegModel` restores punctuation, true-cases words (including acronyms such as `U.S.`), and splits sentences, all in a single forward pass.

Its main use is cleaning up speech-to-text output. CTC recognizers such as Wav2Vec2 return upper-case text with no punctuation. That text is hard to read, and cased models such as [NER](named-entity-recognition.md) perform poorly on it.

## Quick example

```java
try (var model = PunctCapSegModel.builder().build()) {
    List<String> sentences = model.infer(
        "marie curie moved from poland to paris and later worked with the us radium institute "
        + "she won two nobel prizes");
    // ["Marie Curie moved from Poland to Paris and later worked with the U.S. Radium Institute.",
    //  "She won two Nobel Prizes."]
}
```

## With speech recognition

Wrap any `SpeechRecognizer` in `PunctuatedSpeechRecognizer` to get formatted transcripts directly:

```java
try (SpeechRecognizer recognizer = new PunctuatedSpeechRecognizer(
        Wav2Vec2Recognizer.builder().build(),
        PunctCapSegModel.builder().build())) {
    String text = recognizer.transcribe(Path.of("speech.wav")).text();
    // Wav2Vec2 alone:  "MARIE CURIE MOVED TO PARIS"
    // Punctuated:      "Marie Curie moved to Paris."
}
```

The decorator formats only `Transcription.text()`. Segments pass through unchanged. Closing it closes both the wrapped recognizer and the model.

## Full example

```java
import io.github.inference4j.nlp.PunctCapSegModel;
import java.util.List;

public class PunctuationExample {
    public static void main(String[] args) {
        try (var model = PunctCapSegModel.builder().build()) {
            List<String> sentences = model.infer("MARIE CURIE MOVED TO PARIS SHE WON TWO NOBEL PRIZES");
            sentences.forEach(System.out::println);
        }
    }
}
```

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | `inference4j/punctuation-fullstop-truecase-english` | HuggingFace model ID |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.sessionOptions(SessionConfigurer)` | `SessionConfigurer` | default | ONNX Runtime session config |
| `.tokenizer(UnigramTokenizer)` | `UnigramTokenizer` | auto-loaded from `tokenizer.json` | Custom tokenizer |
| `.maxLength(int)` | `int` | `256` | Maximum tokens per call, including the begin/end markers |
| `.truncation(TruncationPolicy)` | `TruncationPolicy` | `TRUNCATE` | Input longer than the token limit: `TRUNCATE` keeps the first tokens and logs a warning, `FAIL` throws `InputTooLongException`. See [Input length](../reference/configuration.md#input-length) |

## Result type

`infer(String)` returns a `List<String>` with one entry per detected sentence, in order. Blank input returns an empty list.

## Input length

| Limit | Behavior when exceeded | Long-input strategies |
|-------|------------------------|-----------------------|
| 256 tokens (about 200 words), including begin/end markers | Truncated with a warning (default), or rejected with `.truncation(TruncationPolicy.FAIL)` | None yet |

Keep inputs short. When formatting transcripts of long audio, transcribe in pieces, for example one call per [voice activity](voice-activity-detection.md) segment, rather than formatting a whole recording at once.

## Available models

| Model | Wrapper | Size | License |
|-------|---------|------|---------|
| `inference4j/punctuation-fullstop-truecase-english` | `PunctCapSegModel` | ~210 MB | Apache 2.0 |

Mirrored from [1-800-BAD-CODE/punctuation_fullstop_truecase_english](https://huggingface.co/1-800-BAD-CODE/punctuation_fullstop_truecase_english). The class name mirrors `PunctCapSegModelONNX` from the upstream [punctuators](https://github.com/1-800-BAD-CODE/punctuators) package.

## How it works

1. The input is lower-cased, stripped of punctuation (apostrophes are kept), and whitespace-collapsed. Already-formatted text is therefore re-formatted, never double-punctuated.
2. The text is tokenized with a SentencePiece Unigram tokenizer and wrapped in begin/end markers.
3. A single forward pass produces three predictions per token:
    - **casing**: which characters of the token to upper-case
    - **punctuation**: none, `.`, `,`, `?`, or *acronym*, where a period follows every letter (`us` → `U.S.`)
    - **sentence boundary**: whether a sentence ends after this token
4. The tokens are reassembled into text and split into sentences.

## Tips

- English only. Supported punctuation is `.`, `,` and `?`.
- Numbers stay as the input wrote them. The model does not convert spoken numbers (`FIVE FIVE FIVE`) into digits.
- A character outside the model's vocabulary appears as `<unk>` in the output.
- Not thread-safe. Use one instance per thread.
