# Whisper Speech-to-Text

Transcribe and translate speech using OpenAI's Whisper models, with automatic chunking for long audio.

See the [overview](introduction.md) for background on how autoregressive generation differs from single-pass inference.

## Models

| Model ID | Parameters | Size | 11 s clip on CPU* |
|----------|------------|------|-------------------|
| `inference4j/whisper-tiny-genai` | 39M | ~225 MB | ~0.1 s |
| `inference4j/whisper-small-genai` | 244M | ~1.1 GB | ~0.5 s |

\* Measured on an Apple M4 Pro. Both are FP32 exports made with onnxruntime-genai's model builder,
and download on first use.

`small` transcribes noticeably better, including punctuation such as commas; `tiny` is the
choice when size and speed matter most.

!!! note "Verified for English transcription"

    Our tests cover English transcription. Whisper also supports 98 other languages and
    translation to English (shown below), but we have not verified those yet.

## Quick example

```java
try (var whisper = WhisperSpeechModel.builder()
        .modelId("inference4j/whisper-small-genai")
        .build()) {
    System.out.println(whisper.transcribe(Path.of("meeting.wav")).text());
}
```

## Full example

```java
import io.github.inference4j.genai.audio.WhisperSpeechModel;
import io.github.inference4j.audio.Transcription;
import java.nio.file.Path;

public class WhisperTranscription {
    public static void main(String[] args) {
        try (var whisper = WhisperSpeechModel.builder()
                .modelId("inference4j/whisper-small-genai")
                .build()) {

            Transcription result = whisper.transcribe(Path.of("meeting.wav"));
            System.out.println(result.text());
        }
    }
}
```

## Translation

Whisper can translate speech from any supported language into English:

```java
try (var whisper = WhisperSpeechModel.builder()
        .modelId("inference4j/whisper-small-genai")
        .language("fr")
        .task(WhisperTask.TRANSLATE)
        .build()) {
    Transcription result = whisper.transcribe(Path.of("french-audio.wav"));
    System.out.println(result.text());  // English translation
}
```

## From raw audio data

If you already have audio samples as a float array:

```java
try (var whisper = WhisperSpeechModel.builder()
        .modelId("inference4j/whisper-small-genai")
        .build()) {
    float[] samples = loadAudioSamples();
    Transcription result = whisper.transcribe(samples, 16000);
    System.out.println(result.text());
}
```

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.modelId(String)` | `String` | — | HuggingFace model ID (required) |
| `.modelSource(ModelSource)` | `ModelSource` | `HuggingFaceModelSource` | Model resolution strategy |
| `.language(String)` | `String` | `"en"` | Source language code (e.g., `"fr"`, `"de"`, `"ja"`) |
| `.task(WhisperTask)` | `WhisperTask` | `TRANSCRIBE` | `TRANSCRIBE` or `TRANSLATE` (to English) |
| `.maxLength(int)` | `int` | `448` | Maximum number of tokens to generate per chunk |
| `.temperature(double)` | `double` | `0.0` | Sampling temperature (0 = greedy) |
| `.topK(int)` | `int` | `0` (disabled) | Top-K sampling |
| `.topP(double)` | `double` | `0.0` (disabled) | Nucleus sampling |

## Result type

`Transcription` is a record with:

| Field | Type | Description |
|-------|------|-------------|
| `text()` | `String` | The transcribed or translated text |
| `segments()` | `List<Segment>` | Timed segments — always empty for Whisper, which runs without timestamps |

Each `Segment` contains:

| Field | Type | Description |
|-------|------|-------------|
| `text()` | `String` | Segment text |
| `startTime()` | `float` | Start time in seconds |
| `endTime()` | `float` | End time in seconds |

## Auto-chunking

Whisper processes audio in 30-second windows. For audio longer than 30 seconds,
`WhisperSpeechModel` automatically:

1. Splits the audio into 30-second chunks
2. Transcribes each chunk independently
3. Concatenates the results

This happens transparently — just call `transcribe()` with any length audio.

!!! warning "Chunks are cut at fixed 30-second boundaries"

    A cut can land in the middle of a word or sentence, and Whisper may then repeat or garble
    a few words around the boundary. Cutting at pauses instead is tracked in
    [#86](https://github.com/inference4j/inference4j/issues/86).

## How it works

Unlike [Wav2Vec2](../use-cases/speech-to-text.md) (single-pass CTC), Whisper is an
**autoregressive** encoder-decoder model. The audio is encoded into a mel spectrogram,
then the decoder generates text tokens one at a time.

```mermaid
flowchart LR
    A["Audio file"] --> B["onnxruntime-genai<br>mel spectrogram → encoder → decoder loop"]
    B --> C["Transcription"]
```

All heavy lifting — mel spectrogram computation, encoder forward pass, autoregressive
decoding and the KV cache — is handled natively by onnxruntime-genai's C++ layer.

## Whisper vs Wav2Vec2

| | Wav2Vec2 | Whisper |
|---|---------|---------|
| Architecture | CTC (single-pass) | Encoder-decoder (autoregressive) |
| Speed | Fast | Slower (token-by-token) |
| Accuracy | Good on clean speech | Better on diverse audio |
| Languages | English (default model) | 99 languages |
| Translation | No | Yes (to English) |
| Module | `inference4j-core` | `inference4j-genai` |

## Tips

- Use `WhisperTask.TRANSLATE` to translate any language to English in a single step.
- Smaller models (tiny, base) are faster but less accurate. The `small` model is a good balance.
- `temperature(0.0)` (default) gives deterministic, greedy decoding — best for transcription accuracy.
- Reuse `WhisperSpeechModel` instances — each one holds the model in memory.
- For short, clean English audio where speed matters, [Wav2Vec2](../use-cases/speech-to-text.md) may be a better fit.
