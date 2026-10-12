# Text Generation

Generate text with decoder-only language models like Phi-4-mini, Qwen3, SmolLM3 and DeepSeek-R1, with built-in streaming support.

See the [overview](introduction.md) for background on how autoregressive generation differs from single-pass inference.

## Quick example

```java
try (var generator = TextGenerator.builder()
        .model(ModelSources.phi3Mini())
        .build()) {
    System.out.println(generator.generate("What is Java?").text());
}
```

## Models

| Preset | Model | Size (INT4) | License | Notes |
|--------|-------|-------------|---------|-------|
| `ModelSources.qwen3_0_6B()` | Qwen3-0.6B | ~0.4 GB | Apache-2.0 | Smallest; multilingual |
| `ModelSources.deepSeekR1_1_5B()` | DeepSeek-R1-Distill-Qwen-1.5B | ~1 GB | MIT | Reasoning: writes its thinking before the answer |
| `ModelSources.smolLM3_3B()` | SmolLM3-3B | ~2 GB | Apache-2.0 | |
| `ModelSources.phi3Mini()` | Phi-3-mini-4k-instruct | ~2.7 GB | MIT | |
| `ModelSources.phi4Mini()` | Phi-4-mini-instruct | ~4.9 GB | MIT | 128K-token context |

Qwen3 and SmolLM3 can also "think" (write out their reasoning in a `<think>` block before
answering). Their presets use the models' **non-thinking mode**, so they answer directly. For
thinking mode, pass your own `ChatTemplate` without the empty `<think></think>` block (and, for
SmolLM3, with `/think` in the system prompt).

## Full example

```java
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.genai.ModelSources;
import io.github.inference4j.genai.nlp.TextGenerator;

public class TextGeneration {
    public static void main(String[] args) {
        try (var generator = TextGenerator.builder()
                .model(ModelSources.phi3Mini())
                .maxLength(200)
                .temperature(0.7)
                .build()) {

            GenerationResult result = generator.generate("What is Java in one sentence?");

            System.out.println(result.text());
            System.out.printf("%d tokens in %,d ms%n",
                    result.generatedTokens(), result.duration().toMillis());
        }
    }
}
```

<figure markdown="span">
  ![Screenshot from showcase app](../assets/images/text-generation.png)
  <figcaption>Screenshot from showcase app</figcaption>
</figure>

## Streaming

Pass a `Consumer<String>` to receive tokens as they are generated:

```java
try (var generator = TextGenerator.builder()
        .model(ModelSources.deepSeekR1_1_5B())
        .maxLength(200)
        .build()) {
    generator.generate("Explain recursion in simple terms.", token -> System.out.print(token));
}
```

The final `GenerationResult` is still returned after generation completes, containing
the full text and timing information.

## Builder options

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `.model(GenerativeModel)` | `GenerativeModel` | — | Preconfigured model from `ModelSources` |
| `.modelSource(ModelSource)` | `ModelSource` | — | Custom model source (requires `.chatTemplate()`) |
| `.chatTemplate(ChatTemplate)` | `ChatTemplate` | — | Prompt formatting for custom models |
| `.maxLength(int)` | `int` | `1024` | Maximum sequence length in tokens, prompt plus generated text. onnxruntime-genai reserves memory for the key/value cache up front based on this value (see [Tips](#tips)) |
| `.temperature(double)` | `double` | `1.0` | Sampling temperature (higher = more random) |
| `.topK(int)` | `int` | `0` (disabled) | Top-K sampling (keep K most probable tokens) |
| `.topP(double)` | `double` | `0.0` (disabled) | Nucleus sampling (keep tokens summing to P probability) |

## Result type

`GenerationResult` is a record with three fields:

| Field | Type | Description |
|-------|------|-------------|
| `text()` | `String` | The generated text |
| `promptTokens()` | `int` | Number of tokens in the input prompt (0 if unknown) |
| `generatedTokens()` | `int` | Number of tokens generated |
| `duration()` | `Duration` | Wall-clock generation time |

## How it works

`TextGenerator` formats the prompt using the model's chat template, then delegates to
onnxruntime-genai for tokenization, the autoregressive generation loop, and decoding.

```mermaid
flowchart LR
    A["User prompt"] --> B["ChatTemplate<br>format prompt"]
    B --> C["onnxruntime-genai<br>tokenize → generate → decode"]
    C --> D["GenerationResult"]
```

See the [overview](introduction.md) for a detailed explanation of the
generation loop, KV cache, and why this architecture differs from single-pass wrappers.

## Tips

- Generation speed scales with model size. DeepSeek-R1 (1.5B) is noticeably faster
  than Phi-3 (3.8B) on the same hardware.
- Keep `maxLength` close to what you need. It bounds the prompt plus the answer, and
  onnxruntime-genai **allocates the key/value cache for the full `maxLength` up front**. For
  SmolLM3-3B we measured a peak of 2.8 GB at `maxLength(256)`, but 12.3 GB at its full
  65,536-token context.
- `temperature` below 1.0 gives more focused output; above 1.0 gives more varied output.
- Reuse `TextGenerator` instances across prompts — each one holds the model in memory.
