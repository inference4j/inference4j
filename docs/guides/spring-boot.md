# Spring Boot

inference4j provides a Spring Boot starter with opt-in auto-configuration for all supported models.

## Setup

Add the starter dependency:

=== "Gradle"

    ```groovy
    implementation 'io.github.inference4j:inference4j-spring-boot-starter:${inference4jVersion}'
    ```

=== "Maven"

    ```xml
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-spring-boot-starter</artifactId>
        <version>${inference4jVersion}</version>
    </dependency>
    ```

## Configuration

Every model is **opt-in** — nothing is downloaded until you set `enabled: true`.

```yaml
inference4j:
  nlp:
    text-classifier:
      enabled: true
```

### All properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `inference4j.nlp.text-classifier.enabled` | `boolean` | `false` | Enable DistilBERT text classifier |
| `inference4j.nlp.text-classifier.model-id` | `String` | `inference4j/distilbert-base-uncased-finetuned-sst-2-english` | Model ID |
| `inference4j.nlp.text-embedder.enabled` | `boolean` | `false` | Enable SentenceTransformer embedder |
| `inference4j.nlp.text-embedder.model-id` | `String` | *(required)* | Model ID (no default — must be specified) |
| `inference4j.nlp.search-reranker.enabled` | `boolean` | `false` | Enable MiniLM cross-encoder reranker |
| `inference4j.nlp.search-reranker.model-id` | `String` | `inference4j/ms-marco-MiniLM-L-6-v2` | Model ID |
| `inference4j.vision.image-classifier.enabled` | `boolean` | `false` | Enable ResNet image classifier |
| `inference4j.vision.image-classifier.model-id` | `String` | `inference4j/resnet50-v1-7` | Model ID |
| `inference4j.vision.object-detector.enabled` | `boolean` | `false` | Enable YOLOv8 object detector |
| `inference4j.vision.object-detector.model-id` | `String` | `inference4j/yolov8n` | Model ID |
| `inference4j.vision.text-detector.enabled` | `boolean` | `false` | Enable CRAFT text detector |
| `inference4j.vision.text-detector.model-id` | `String` | `inference4j/craft-mlt-25k` | Model ID |
| `inference4j.audio.speech-recognizer.enabled` | `boolean` | `false` | Enable Wav2Vec2 speech recognizer |
| `inference4j.audio.speech-recognizer.model-id` | `String` | `inference4j/wav2vec2-base-960h` | Model ID |
| `inference4j.audio.vad.enabled` | `boolean` | `false` | Enable Silero VAD |
| `inference4j.audio.vad.model-id` | `String` | `inference4j/silero-vad` | Model ID |

## Usage

Beans are registered by their **interface type**, so you inject the interface — not the concrete implementation:

```java
@RestController
public class SentimentController {
    private final TextClassifier classifier;

    public SentimentController(TextClassifier classifier) {
        this.classifier = classifier;
    }

    @PostMapping("/analyze")
    public List<TextClassification> analyze(@RequestBody String text) {
        return classifier.classify(text);
    }
}
```

## Overriding beans

All auto-configured beans use `@ConditionalOnMissingBean`, so you can replace any model with your own implementation:

```java
@Configuration
public class CustomModelConfig {

    @Bean
    public TextClassifier textClassifier() {
        return DistilBertTextClassifier.builder()
            .modelId("your-org/your-model")
            .sessionOptions(opts -> opts.addCoreML())
            .build();
    }
}
```

When you define your own bean, the auto-configured one is skipped.

## Health indicator

An actuator health indicator is included automatically when Spring Boot Actuator is on the classpath. It reports the number of registered `InferenceTask` beans.

```
GET /actuator/health
```

```json
{
  "status": "UP",
  "components": {
    "inference4j": {
      "status": "UP",
      "details": {
        "tasks": 3
      }
    }
  }
}
```

The health indicator is enabled by default. Disable it with:

```yaml
inference4j:
  health:
    enabled: false
```

## Example configuration

A typical production setup enabling sentiment analysis and semantic search:

```yaml
inference4j:
  nlp:
    text-classifier:
      enabled: true
    text-embedder:
      enabled: true
      model-id: inference4j/all-MiniLM-L6-v2
    search-reranker:
      enabled: true
```

```java
@Service
public class SearchService {
    private final TextEmbedder embedder;
    private final SearchReranker reranker;

    public SearchService(TextEmbedder embedder, SearchReranker reranker) {
        this.embedder = embedder;
        this.reranker = reranker;
    }

    public List<SearchResult> search(String query, List<String> candidates) {
        // Stage 1: embed and retrieve
        float[] queryEmb = embedder.encode(query);
        // ... cosine similarity ranking ...

        // Stage 2: rerank top candidates
        float[] scores = reranker.scoreBatch(query, topCandidates);
        // ... sort by score ...
        return results;
    }
}
```

## Lazy loading

All inference4j beans are **lazy by default** — models are downloaded and ONNX sessions are created on first use, not at application startup. This keeps startup fast even when multiple models are enabled.

The trade-off is that the first request to each model may be slower due to the model download and session initialization.

### Warming up specific models

If you need a model ready immediately (e.g., for latency-sensitive endpoints), inject it eagerly at startup with an `ApplicationReadyEvent` listener:

```java
@Component
public class ModelWarmup {
    private final TextClassifier classifier;

    public ModelWarmup(TextClassifier classifier) {
        this.classifier = classifier;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void warmup() {
        classifier.classify("warmup");
    }
}
```

This triggers the lazy bean initialization during startup, so the model is ready when the first real request arrives.

## Tips

- Use `model-id` to swap models without changing code (e.g., switch from ResNet to EfficientNet).
- The text embedder has no default model ID — you must specify one via `model-id`.
- All beans implement `AutoCloseable` and are properly cleaned up on application shutdown.
