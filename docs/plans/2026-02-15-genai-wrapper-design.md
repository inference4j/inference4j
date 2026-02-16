# Autoregressive Generation Wrapper Design

## Goal

Wrap onnxruntime-genai to bring autoregressive text generation into inference4j. Same builder conventions, same 3-line usage, same domain-package structure — but backed by genai's native generate loop instead of single-pass ONNX Runtime inference.

## Context

All current inference4j models are **single-pass**: one forward pass, one result. Autoregressive models produce output **token-by-token**, where each token depends on all previous tokens. This includes LLMs (Phi-3, GPT-2), speech-to-text with attention (Whisper), and vision-language models (Phi-3.5 vision).

[onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) provides the generate loop with KV cache management, sampling strategies, and streaming — built on top of ONNX Runtime. We published [community Java bindings](https://github.com/inference4j/onnxruntime-genai) to Maven Central (`io.github.inference4j:onnxruntime-genai:0.12.0`) and validated them end-to-end with Phi-3-mini (70.5 tokens/sec on CPU).

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Module | New `inference4j-genai` | Clean separation — single-pass users never pull in genai native libs |
| Base class | `AbstractGenerativeTask<I,O>` | Parallel hierarchy to `AbstractInferenceTask<I,O>`. Owns the generate loop. |
| Does NOT extend `AbstractInferenceTask` | Correct | Genai handles its own tokenization and generation loop. No pre/infer/post pipeline. |
| Generation params | Builder time | Baked into the instance. Matches inference4j convention (configure once, use many times). |
| Prompt formatting | `Tokenizer.applyChatTemplate()` | Delegates to model's built-in Jinja2 template. Zero model-specific configuration. |
| Model resolution | Extend `HuggingFaceModelSource` | Add `resolve(repoId, subdirectory)` for downloading genai model directories. Reuses existing caching/locking. |
| Return type | `GenerationResult` record | Consistent with `Transcription`, `Classification`, `Detection` pattern. |
| Domain packages | Same as single-pass wrappers | `io.github.inference4j.nlp`, `.audio`, `.vision`. Packages = domain, modules = deployment. |
| Phase 1 scope | `TextGenerator` only | Prove the pattern, then add Whisper and vision-language. |

## Architecture

### Why not AbstractInferenceTask?

| | Single-pass (current) | Generative (genai) |
|---|---|---|
| Pipeline | pre → infer → post | tokenize → generate loop (N steps) → decode |
| Session | `InferenceSession` (our wrapper) | `Model` (genai native handle) |
| Tokenization | inference4j owns it | genai owns it (bundled in model dir) |
| Output | One result immediately | Tokens stream one at a time |

Forcing genai into `AbstractInferenceTask` with `Preprocessor`/`Postprocessor` would fight the library. The generate loop **is** the inference.

### Type hierarchy

```
Single-pass:  InferenceTask<I,O> → AbstractInferenceTask<I,O> → concrete wrappers
Generative:   GenerativeTask<I,O> → AbstractGenerativeTask<I,O> → concrete wrappers
```

### Multimodal support

All genai models produce text output, but accept different inputs:

- **Text LLMs** (Phi-3, GPT-2): `String → GenerationResult`
- **Speech-to-text** (Whisper): `Path → Transcription`
- **Vision-language** (Phi-3.5 vision): `ImagePrompt → GenerationResult`

The generate loop is identical for all three. The difference is **how the generator gets fed**. The `prepareGenerator()` hook handles this:

```java
// Text: encode prompt → append token sequences
void prepareGenerator(String input, Generator gen) {
    gen.appendTokenSequences(tokenizer.encode(formatPrompt(input)));
}

// Audio: process audio → set named tensors
void prepareGenerator(Path audio, Generator gen) {
    NamedTensors inputs = processor.processAudios("", new Audios(audio.toString()));
    gen.setInputs(inputs);
}

// Vision: process image + prompt → set named tensors
void prepareGenerator(ImagePrompt input, Generator gen) {
    NamedTensors inputs = processor.processImages(input.prompt(), new Images(input.imagePath()));
    gen.setInputs(inputs);
}
```

## Module layout

```
inference4j-genai/
  build.gradle
  src/main/java/io/github/inference4j/
    genai/
      GenerativeTask.java             — interface
      AbstractGenerativeTask.java     — base class with generate loop
      GenerationResult.java           — result record
    nlp/
      TextGenerator.java              — Phase 1: Phi-3, GPT-2, Llama, etc.
    audio/
      WhisperTranscriber.java         — Phase 2: Whisper speech-to-text
    vision/
      VisionLanguageGenerator.java    — Phase 3: Phi-3.5 vision, captioning
```

Dependencies:

```groovy
dependencies {
    api 'io.github.inference4j:onnxruntime-genai:0.12.0'
    implementation project(':inference4j-core')  // ModelSource, HuggingFaceModelSource
}
```

Depends on `inference4j-core` for model resolution but **not** on `inference4j-preprocessing` or `inference4j-tasks`. Genai handles its own tokenization natively.

## API design

### GenerativeTask interface

```java
public interface GenerativeTask<I, O> extends AutoCloseable {
    O generate(I input);
    O generate(I input, Consumer<String> tokenListener);
}
```

### GenerationResult record

```java
public record GenerationResult(
    String text,
    int tokenCount,
    long durationMillis
) {}
```

### AbstractGenerativeTask base class

```java
public abstract class AbstractGenerativeTask<I, O> implements GenerativeTask<I, O> {

    protected final Model model;
    protected final Tokenizer tokenizer;
    private final int maxLength;
    private final double temperature;
    private final int topK;
    private final double topP;

    @Override
    public final O generate(I input) {
        return generate(input, null);
    }

    @Override
    public final O generate(I input, Consumer<String> tokenListener) {
        try (GeneratorParams params = createParams();
             Generator generator = new Generator(model, params)) {

            prepareGenerator(input, generator);

            long start = System.currentTimeMillis();
            int tokenCount = 0;
            StringBuilder sb = new StringBuilder();

            try (TokenizerStream stream = tokenizer.createStream()) {
                while (!generator.isDone()) {
                    generator.generateNextToken();
                    int token = generator.getLastTokenInSequence(0);
                    String text = stream.decode(token);
                    sb.append(text);
                    tokenCount++;
                    if (tokenListener != null && !text.isEmpty()) {
                        tokenListener.accept(text);
                    }
                }
            }

            long duration = System.currentTimeMillis() - start;
            return parseOutput(sb.toString(), input, tokenCount, duration);
        }
    }

    protected abstract void prepareGenerator(I input, Generator generator);

    protected abstract O parseOutput(String generatedText, I input,
                                     int tokenCount, long durationMillis);

    private GeneratorParams createParams() {
        GeneratorParams params = new GeneratorParams(model);
        params.setSearchOption("max_length", maxLength);
        if (temperature > 0) params.setSearchOption("temperature", temperature);
        if (topK > 0) params.setSearchOption("top_k", topK);
        if (topP > 0) params.setSearchOption("top_p", topP);
        return params;
    }

    @Override
    public void close() {
        tokenizer.close();
        model.close();
    }
}
```

### TextGenerator (Phase 1)

```java
public class TextGenerator extends AbstractGenerativeTask<String, GenerationResult> {

    private static final String DEFAULT_MODEL_ID =
        "microsoft/Phi-3-mini-4k-instruct-onnx";
    private static final String DEFAULT_SUBDIRECTORY =
        "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4";

    @Override
    protected void prepareGenerator(String input, Generator generator) {
        String formatted = tokenizer.applyChatTemplate(
            null,  // use model's default template
            buildMessagesJson(input),
            null,  // no tools
            true   // add generation prompt
        );
        Sequences sequences = tokenizer.encode(formatted);
        generator.appendTokenSequences(sequences);
    }

    @Override
    protected GenerationResult parseOutput(String text, String input,
                                           int tokenCount, long durationMillis) {
        return new GenerationResult(text, tokenCount, durationMillis);
    }

    private String buildMessagesJson(String userMessage) {
        return "[{\"role\": \"user\", \"content\": \"%s\"}]"
            .formatted(userMessage.replace("\"", "\\\""));
    }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private String modelId = DEFAULT_MODEL_ID;
        private String subdirectory = DEFAULT_SUBDIRECTORY;
        private ModelSource modelSource;
        private int maxLength = 1024;
        private double temperature = 1.0;
        private int topK = 0;
        private double topP = 0.0;

        public Builder modelId(String modelId) { ... }
        public Builder subdirectory(String subdirectory) { ... }
        public Builder modelSource(ModelSource modelSource) { ... }
        public Builder maxLength(int maxLength) { ... }
        public Builder temperature(double temperature) { ... }
        public Builder topK(int topK) { ... }
        public Builder topP(double topP) { ... }

        public TextGenerator build() {
            ModelSource source = modelSource != null
                ? modelSource : HuggingFaceModelSource.defaultInstance();
            Path modelDir = source.resolve(modelId, subdirectory);
            Model model = new Model(modelDir.toString());
            Tokenizer tokenizer = new Tokenizer(model);
            return new TextGenerator(model, tokenizer, maxLength,
                                     temperature, topK, topP);
        }
    }
}
```

### Usage

```java
// 3-line hero snippet
try (var gen = TextGenerator.builder().build()) {
    GenerationResult result = gen.generate("What is Java in one sentence?");
    System.out.println(result.text());
}

// With streaming
try (var gen = TextGenerator.builder()
        .maxLength(200)
        .temperature(0.7)
        .build()) {
    gen.generate("Explain recursion.", token -> System.out.print(token));
}

// Custom model
try (var gen = TextGenerator.builder()
        .modelId("microsoft/Phi-3.5-mini-instruct-onnx")
        .subdirectory("cpu/cpu-int4-rtn-block-32-acc-level-4")
        .maxLength(500)
        .build()) {
    GenerationResult result = gen.generate("Write a haiku about Java.");
    System.out.printf("%s%n[%d tokens in %dms]%n",
        result.text(), result.tokenCount(), result.durationMillis());
}
```

## Model resolution changes

`HuggingFaceModelSource` needs a new method for genai models:

```java
// Existing: download specific files
Path resolve(String repoId, List<String> requiredFiles);

// New: download all files under a subdirectory
Path resolve(String repoId, String subdirectory);
```

The new method lists files in the subdirectory via the HuggingFace API (`/api/models/{repoId}/tree/main/{subdirectory}`), then downloads each one using the existing atomic-download infrastructure.

## Phased delivery

### Phase 1: TextGenerator
- `GenerativeTask`, `AbstractGenerativeTask`, `GenerationResult`
- `TextGenerator` with builder
- `HuggingFaceModelSource.resolve(repoId, subdirectory)`
- Unit tests + model integration test with Phi-3-mini
- Example in `inference4j-examples`

### Phase 2: Whisper
- `WhisperTranscriber` in `io.github.inference4j.audio`
- Uses `MultiModalProcessor.processAudios()`
- Returns `Transcription` (reuse existing type from Wav2Vec2)
- Mel spectrogram handled natively by genai (no Java preprocessing needed)

### Phase 3: Vision-language
- `VisionLanguageGenerator` in `io.github.inference4j.vision`
- Uses `MultiModalProcessor.processImages()`
- New `ImagePrompt` record: `(Path imagePath, String prompt)`
- Returns `GenerationResult`

## Testing strategy

- **Unit tests**: Mock `Model`/`Generator`/`Tokenizer` via package-private constructor injection. Test `prepareGenerator()`, `parseOutput()`, builder validation.
- **Model integration tests**: `./gradlew modelTest` downloads Phi-3-mini and runs real generation. Verify output is non-empty, coherent, and within token/time bounds.
