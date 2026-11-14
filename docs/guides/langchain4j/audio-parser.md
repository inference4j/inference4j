# Audio document parser

`Inference4jAudioDocumentParser` is a LangChain4j `DocumentParser` that turns a recording into text documents with
inference4j [speech-to-text](../../use-cases/speech-to-text.md), in-process. Segments carry their start and end time in
the recording.

## Quick example

```java
try (var parser = new Inference4jAudioDocumentParser(new SegmentedSpeechRecognizer(
        MoonshineRecognizer.builder().build(),
        SileroVadDetector.builder().build()))) {

    // One document with the whole transcript
    Document transcript = FileSystemDocumentLoader.loadDocument(Path.of("meeting.wav"), parser);

    // One document per timed segment
    try (InputStream in = Files.newInputStream(Path.of("meeting.wav"))) {
        List<Document> segments = parser.parseSegments(in);
    }
}
```

`SegmentedSpeechRecognizer` cuts long recordings at pauses and produces the timed segments. Whisper requires it
for audio longer than 30 seconds.

## Two ways to parse

| Method | Returns | Use it for |
|--------|---------|-----------|
| `parse(InputStream)` | One `Document` with the whole transcript, no timing metadata | LangChain4j document loaders; the transcript is split by a document splitter |
| `parseSegments(InputStream)` | One `Document` per segment, in recording order, with timing metadata | Retrieval with the time of each segment |

`parseSegments` is not part of LangChain4j's `DocumentParser` contract (one stream, one document), so call it
directly and pass the result to the ingestor:

```java
try (InputStream in = Files.newInputStream(Path.of("meeting.wav"))) {
    EmbeddingStoreIngestor.builder()
            .embeddingModel(embeddingModel)
            .embeddingStore(store)
            .build()
            .ingest(parser.parseSegments(in));
}
```

Without a document splitter, the ingestor stores each segment document as one text segment, with its metadata.

### Segment metadata

| Key | Type | Value |
|-----|------|-------|
| `segment_index` (`Inference4jAudioDocumentParser.SEGMENT_INDEX`) | `int` | Position of the segment in the recording, from 0 |
| `start_ms` (`Inference4jAudioDocumentParser.START_MS`) | `long` | Where the segment starts, in milliseconds |
| `end_ms` (`Inference4jAudioDocumentParser.END_MS`) | `long` | Where the segment ends, in milliseconds |

An `InputStream` has no file name. To identify the recording, add a metadata key:

```java
List<Document> segments = parser.parseSegments(in);
segments.forEach(segment -> segment.metadata().put("file_name", "meeting.wav"));
```

The metadata is returned with the retrieved content:

```java
interface Assistant {
    Result<String> chat(String message);
}

Result<String> result = assistant.chat("What did we decide about the launch date?");
for (Content source : result.sources()) {
    long startMs = source.textSegment().metadata().getLong(Inference4jAudioDocumentParser.START_MS);
    System.out.printf("%s at %d:%02d%n",
            source.textSegment().metadata().getString("file_name"), startMs / 60_000, startMs / 1000 % 60);
}
```

## Choosing a recognizer

The parser has one constructor, `Inference4jAudioDocumentParser(SpeechRecognizer)`, and accepts any inference4j
recognizer.

| Recognizer | Module | Size | Output | Notes |
|------------|--------|------|--------|-------|
| `MoonshineRecognizer` | `inference4j-core` | ~140 MB (small, default) | Punctuated, cased | English |
| `WhisperSpeechModel` | `inference4j-genai` | ~225 MB (tiny) to ~1.1 GB (small) | Punctuated, cased | 99 languages; must be wrapped in `SegmentedSpeechRecognizer` for audio over 30 seconds |
| `Wav2Vec2Recognizer` | `inference4j-core` | ~370 MB | Upper-case, no punctuation | English. Combine with `PunctuatedSpeechRecognizer` to restore punctuation and casing |

```java
// Wav2Vec2 with punctuation restored, cut at pauses
var parser = new Inference4jAudioDocumentParser(new SegmentedSpeechRecognizer(
        new PunctuatedSpeechRecognizer(
                Wav2Vec2Recognizer.builder().build(),
                PunctCapSegModel.builder().build()),
        SileroVadDetector.builder().build()));
```

Segment length is set on `SegmentedSpeechRecognizer`: pieces end at pauses and are at most 30 seconds by default.
Pass a `Duration` as the third constructor argument for shorter segments. See
[Long recordings](../../use-cases/speech-to-text.md#long-recordings).

## Notes

- **WAV only.** Input must be WAV (8- or 16-bit PCM, mono or stereo, any sample rate). Convert MP3, M4A and video
  first, for example `ffmpeg -i talk.mp4 -ac 1 -ar 16000 talk.wav`. Other input fails with `InferenceException`.
- **Silence.** `parse` throws `InferenceException` ("No speech detected in audio") because LangChain4j documents
  cannot be blank. `parseSegments` returns an empty list.
- **Segment boundaries.** A segment spans one piece of audio between pauses, including
  up to half a pause of silence at each end. There is no speaker identification.
- **Without `SegmentedSpeechRecognizer`**, `parseSegments` returns a single document spanning the whole recording.
- **Memory and disk.** The stream is copied to a temporary file, which is deleted afterwards, and the decoded audio
  is held in memory while it is transcribed.
- **Lifecycle.** `close()` closes the recognizer and, through `SegmentedSpeechRecognizer`, the models it wraps. See
  [Lifecycle](index.md#lifecycle).

inference4j does not implement LangChain4j's `AudioTranscriptionModel` interface, whose response carries text
without timestamps.
