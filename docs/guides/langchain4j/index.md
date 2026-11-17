# inference4j for LangChain4j

`inference4j-langchain4j` provides inference4j models as implementations of LangChain4j interfaces, for use in
ingestion pipelines, retrievers and AI Services. The models run inside the JVM, without an API key, a model server
or Python.

## Adapters

| Use case | Adapter | LangChain4j interface | Page |
|---|---|---|---|
| Redact personal data from documents before they are embedded and stored | `Inference4jPiiRedactor` | `DocumentTransformer`, `TextSegmentTransformer` | [PII redaction](pii-redaction.md) |
| Block prompt injection in user messages with a classifier | `Inference4jPromptInjectionGuardrail` | `InputGuardrail` | [Prompt-injection guardrail](prompt-injection-guardrail.md) |
| Read the text in scanned pages, screenshots and photos | `Inference4jOcrDocumentParser` | `DocumentParser` | [OCR document parser](ocr-parser.md) |
| Transcribe recordings into documents with timestamps | `Inference4jAudioDocumentParser` | `DocumentParser` | [Audio document parser](audio-parser.md) |
| Embed text, with query and document prefixes applied from the input type | `Inference4jEmbeddingModel` | `EmbeddingModel` | [Embedding model](embeddings.md) |
| Rerank retrieved content with a cross-encoder | `Inference4jScoringModel` | `ScoringModel` | [Reranking](reranking.md) |

[RAG pipelines](rag-pipeline.md) shows how the adapters fit together, with examples for recordings and scanned documents.

## Comparison with LangChain4j's built-in options

| Capability | LangChain4j 1.23 | inference4j adapter |
|---|---|---|
| PII detection and redaction | Not provided | GLiNER PII model as a `DocumentTransformer` / `TextSegmentTransformer` |
| Prompt-injection detection | `PatternBasedPromptInjectionGuardrail` (regular expressions), `DecisionModelInputGuardrail` (one LLM call per message) | DeBERTa-v3 classifier, about 10 ms per message on a laptop CPU |
| OCR | Docling parser (calls a separate `docling-serve` process), or a multimodal chat model | PaddleOCR PP-OCRv6 in-process, about 31 MB of models |
| Speech to documents | `AudioTranscriptionModel`, implemented by hosted providers; responses carry no timestamps | Moonshine, Whisper or Wav2Vec2 in-process, one `Document` per timed segment |
| Local embeddings | Packaged in-process ONNX models (all-MiniLM-L6-v2, bge-small, e5-small) and `OnnxEmbeddingModel` for a custom file | bge-base, gte-base, nomic-embed-text-v1.5 (8K context, Matryoshka sizes); query/document prefixes from `EmbeddingInputType` |
| Local reranking | `OnnxScoringModel`, given paths to a model file and a `tokenizer.json` | Cross-encoder downloaded and cached automatically; MaxP for long segments |
| Local chat model | Ollama, Jlama and others | Not provided as a `ChatModel` |

### Choosing an embedding or scoring model

Both libraries run embedding and reranking models in-process on ONNX Runtime.

| | LangChain4j in-process models | inference4j adapters |
|---|---|---|
| Model files | Packaged in the jar (embeddings), or paths supplied by the caller (`OnnxEmbeddingModel`, `OnnxScoringModel`) | Downloaded from HuggingFace on first use and cached |
| Models | all-MiniLM-L6-v2, bge-small-en, bge-small-zh, e5-small-v2, or a custom ONNX file | The models listed under [Embedding model](embeddings.md#model-choices) and [Reranking](reranking.md) |
| Query/document prefixes | Added to the text by the caller | Applied from `EmbeddingInputType` |
| Batches | Embedded in parallel on a thread pool | Embedded one text at a time; each run uses all cores |
| Long input | Embeddings: split into partitions and averaged. Scoring: truncated | Truncated by default; windows with `.stride(int)` (averaged for embeddings, MaxP for scoring) |

## Installation

Add the adapter module next to the LangChain4j dependencies. It includes `inference4j-core`.

=== "Gradle"

    ```groovy
    implementation 'io.github.inference4j:inference4j-langchain4j:${inference4jVersion}'
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
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j</artifactId>
        <version>${langchain4jVersion}</version>
    </dependency>
    ```

| | |
|---|---|
| **inference4j** | 0.14.0 or later |
| **LangChain4j** | 1.23.0 (tested) |
| **Java** | 17 or later |

- LangChain4j is a `compileOnly` dependency of the module. The application supplies its own LangChain4j version.
- ONNX Runtime requires the JVM flag `--enable-native-access=ALL-UNNAMED`. See
  [Installation](../../getting-started/installation.md#jvm-flags).
- `WhisperSpeechModel` requires `inference4j-genai`. All other models on these pages are in `inference4j-core`.

!!! note "Experimental LangChain4j APIs"
    LangChain4j marks guardrails and the embedding request API (`EmbeddingInputType`) as experimental. They may
    change in a later LangChain4j release.

!!! warning "ONNX Runtime version"
    inference4j requires ONNX Runtime 1.28.0. LangChain4j's in-process modules (`langchain4j-embeddings-*`,
    `langchain4j-onnx-scoring`, and `langchain4j-easy-rag`, which includes an embedding module) declare ONNX
    Runtime 1.23.2. When both are on the classpath, make sure the build resolves 1.28.0: Gradle selects the newer
    version, Maven selects the version declared closest to the project. LangChain4j's ONNX modules are not tested
    with 1.28.0.

## Quick start

A retrieval-augmented assistant with a local embedding model and a local guardrail. The chat model can be any
LangChain4j `ChatModel`:

```java
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import io.github.inference4j.langchain4j.Inference4jEmbeddingModel;
import io.github.inference4j.langchain4j.Inference4jPromptInjectionGuardrail;
import io.github.inference4j.nlp.DebertaPromptInjectionDetector;
import io.github.inference4j.nlp.SentenceTransformerEmbedder;

interface Assistant {
    String chat(String message);
}

ChatModel chatModel = ...;          // OpenAI, Anthropic, Ollama: any LangChain4j chat model
List<Document> documents = ...;     // loaded with any LangChain4j document loader

var embeddingModel = Inference4jEmbeddingModel.builder()
        .embedder(SentenceTransformerEmbedder.builder()
                .modelId("inference4j/all-MiniLM-L6-v2").normalize().build())
        .build();

var store = new InMemoryEmbeddingStore<TextSegment>();
EmbeddingStoreIngestor.builder()
        .documentSplitter(DocumentSplitters.recursive(300, 30))
        .embeddingModel(embeddingModel)
        .embeddingStore(store)
        .build()
        .ingest(documents);

Assistant assistant = AiServices.builder(Assistant.class)
        .chatModel(chatModel)
        .contentRetriever(EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store).embeddingModel(embeddingModel).maxResults(4).build())
        .inputGuardrails(new Inference4jPromptInjectionGuardrail(
                DebertaPromptInjectionDetector.builder().build()))
        .build();

assistant.chat("What is the cancellation policy?");
```

Models are downloaded from HuggingFace on first use and cached under `~/.cache/inference4j`. See
[Model loading](../model-loading.md) for offline use and custom locations.

## Lifecycle

!!! danger "Closing an adapter closes its model"
    Each adapter owns the inference4j model passed to it. `close()` on the adapter closes that model and frees its
    native memory.

    - Give each adapter its own model instance. A model wrapped in two adapters is closed when either adapter is
      closed.
    - Calls on a closed model throw `IllegalStateException`.
    - Close adapters on application shutdown. All adapters are `AutoCloseable`.

## Limitations

- **Scanned PDFs.** The OCR parser reads images. PDF pages must be rasterized first; see [OCR document parser](ocr-parser.md#scanned-pdfs).
- **Chat messages.** The PII redactor implements the ingestion interfaces only. See [PII redaction](pii-redaction.md#chat-messages).
- **Compressed audio and video.** The audio parser reads WAV.
- **Chat models.** inference4j's text generators are not available as a LangChain4j `ChatModel`.
