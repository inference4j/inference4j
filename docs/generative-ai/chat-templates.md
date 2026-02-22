# Chat Templates

Every instruction-tuned language model expects prompts in a specific format. The model
was trained on conversations structured with special marker tokens, and it will only
follow instructions reliably if the prompt matches that format. A `ChatTemplate`
tells inference4j how to wrap a user message in the correct markers before passing
it to the tokenizer.

## Why models need different templates

During training, each model family learns to associate specific token sequences with
roles in a conversation. These markers vary across model families:

| Model | User marker | End marker | Assistant marker |
|-------|-------------|------------|------------------|
| Phi-3 | `<\|user\|>` | `<\|end\|>` | `<\|assistant\|>` |
| DeepSeek-R1 (Qwen) | `<\|User\|>` | — | `<\|Assistant\|>` |
| Llama / ChatML | `<\|im_start\|>user` | `<\|im_end\|>` | `<\|im_start\|>assistant` |

If you send a Phi-3 prompt to a DeepSeek model (or vice versa), the model won't
recognize the role markers. It may ignore the instruction entirely, echo the markers
back, or produce incoherent output.

## How it works

`ChatTemplate` is a functional interface with a single method:

```java
@FunctionalInterface
public interface ChatTemplate {
    String format(String userMessage);
}
```

It takes the raw user message and returns the fully formatted prompt string. For Phi-3,
the implementation looks like:

```java
message -> "<|user|>\n" + message + "<|end|>\n<|assistant|>\n"
```

For a prompt like "What is Java?", this produces:

```
<|user|>
What is Java?<|end|>
<|assistant|>
```

The tokenizer then encodes this formatted string into token IDs that the model
recognizes.

## Preconfigured templates

### onnxruntime-genai models

When you use `ModelSources` factory methods, the chat template is already configured:

```java
// Chat template is bundled — nothing extra to configure
TextGenerator.builder()
        .model(ModelSources.phi3Mini())
        .build();
```

`ModelSources.phi3Mini()` returns a `GenerativeModel` that pairs the model source
with the correct Phi-3 chat template. `ModelSources.deepSeekR1_1_5B()` does the same
for DeepSeek's format.

### Native generation models

`OnnxTextGenerator` presets for instruct models (SmolLM2, Qwen2.5) come with a
ChatML template preconfigured. GPT-2 is a base model (not instruction-tuned) so
it has no default template, but you can provide one:

```java
OnnxTextGenerator.gpt2()
        .chatTemplate(msg -> "Q: " + msg + "\nA:")
        .maxNewTokens(100)
        .build();
```

## Custom templates

To use a model that isn't preconfigured in `ModelSources`, provide both a `ModelSource`
and a `ChatTemplate`:

```java
TextGenerator.builder()
        .modelSource(myModelSource)
        .chatTemplate(msg -> "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n")
        .build();
```

The template format for a given model is defined in its `tokenizer_config.json` file
on HuggingFace, under the `chat_template` field. It's typically a Jinja2 template —
you only need to translate the user-message portion into a Java lambda.

## Adding support for a new model

When a new model family uses a chat template that isn't covered by `ModelSources`,
you need two things:

1. **The model files** — a `ModelSource` that resolves the ONNX weights, tokenizer,
   and config files
2. **The chat template** — a `ChatTemplate` lambda that formats prompts with the
   correct markers

Look at the model's `tokenizer_config.json` for the `chat_template` field to find the
expected format. For a simple user message (no system prompt, no tools), the relevant
portion is usually a few lines of Jinja that map to a straightforward string
concatenation in Java.
