# Chat model

`Inference4jChatModel` and `Inference4jStreamingChatModel` are a LangChain4j `ChatModel` and `StreamingChatModel`
backed by an inference4j [`TextGenerator`](../../generative-ai/text-generation.md). The LLM runs inside the JVM
through onnxruntime-genai, without an API key or a model server.

## Quick example

```java
var chatModel = new Inference4jChatModel(TextGenerator.builder()
        .model(ModelSources.phi4Mini())
        .maxNewTokens(300)
        .temperature(0)
        .build());

String answer = chatModel.chat("What is the capital of France?");
```

Phi-4-mini (about 4.9 GB) is downloaded on first use.

## Dependencies

The chat model adapters need `inference4j-genai` next to `inference4j-langchain4j`. The other adapters do not.

=== "Gradle"

    ```groovy
    implementation 'io.github.inference4j:inference4j-langchain4j:${inference4jVersion}'
    implementation 'io.github.inference4j:inference4j-genai:${inference4jVersion}'
    implementation 'dev.langchain4j:langchain4j:${langchain4jVersion}'
    ```

=== "Maven"

    ```xml
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-langchain4j</artifactId>
        <version>${inference4jVersion}</version>
    </dependency>
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-genai</artifactId>
        <version>${inference4jVersion}</version>
    </dependency>
    <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j</artifactId>
        <version>${langchain4jVersion}</version>
    </dependency>
    ```

## With AI Services

```java
interface Assistant {
    @SystemMessage("You are a support assistant. Answer in one or two sentences.")
    String chat(String message);
}

Assistant assistant = AiServices.builder(Assistant.class)
        .chatModel(chatModel)
        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
        .build();
```

System messages, chat memory and retrieved content all reach the model: the whole conversation is formatted with
the model's chat template on every call.

## Streaming

```java
interface StreamingAssistant {
    TokenStream chat(String message);
}

var streamingModel = new Inference4jStreamingChatModel(TextGenerator.builder()
        .model(ModelSources.phi4Mini())
        .maxNewTokens(300)
        .build());

StreamingAssistant assistant = AiServices.builder(StreamingAssistant.class)
        .streamingChatModel(streamingModel)
        .build();

assistant.chat("Name three Java build tools.")
        .onPartialResponse(System.out::print)
        .onCompleteResponse(response -> System.out.println())
        .onError(Throwable::printStackTrace)
        .start();
```

Generation runs on the calling thread: the handler receives every partial response, then the complete response
or the error, before `chat` returns.

## Options

Each adapter has one constructor, taking a `TextGenerator`. Sampling and limits are set on the generator; see its
[builder options](../../generative-ai/text-generation.md#builder-options).

| Generator option | Description |
|------------------|-------------|
| `.model(GenerativeModel)` | A model from `ModelSources`: Phi-3-mini, Phi-4-mini, DeepSeek-R1, Qwen3, SmolLM3 |
| `.maxNewTokens(int)` | Maximum length of each answer |
| `.maxLength(int)` | Tokens for the whole conversation plus the answer; sizes the key/value cache |
| `.temperature(double)` | `0` for greedy decoding, the same answer for the same input |
| `.stopSequence(String)` | Ends the answer when the string appears |
| `.intraOpThreads(int)` | Threads per ONNX Runtime session |

## Response

| Field | Value |
|-------|-------|
| `aiMessage().text()` | The generated text |
| `tokenUsage()` | Prompt tokens, after the chat template is applied, and generated tokens |
| `finishReason()` | `LENGTH` when `maxNewTokens` or `maxLength` was reached, `STOP` otherwise |

## Notes

- **Chat templates.** The conversation is formatted with the generator's `ChatTemplate`. The models in
  `ModelSources` have templates that format whole conversations. A custom model needs a template that implements
  `format(List<ChatMessage>)`; see [Chat templates](../../generative-ai/chat-templates.md#conversations).
- **No tool calling.** A request with tool specifications, or a conversation with tool calls, throws
  `UnsupportedOperationException`. The streaming adapter reports it to `onError`.
- **No per-request sampling.** A request that sets temperature, top-p, top-k, maximum output tokens, stop sequences
  or penalties is rejected. Set them on the generator.
- **Text only.** Images and other non-text content are rejected; a JSON response format is rejected.
- **One call at a time.** The generator runs one generation at a time and queues the others, because each
  generation reserves a key/value cache for `maxLength` tokens. For parallel generation, create one adapter per
  generator.
- **Small models.** Models of a few billion parameters answer focused questions from the given context well. They are
  less reliable at combining facts from several sources or following long instructions.
- **Lifecycle.** `close()` closes the generator. See [Lifecycle](index.md#lifecycle).
