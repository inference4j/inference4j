# Spring Boot

inference4j provides a Spring Boot starter with opt-in auto-configuration for all supported models.

## Requirements

| | |
|---|---|
| **Spring Boot** | 4.0 or later |
| **Spring Framework** | 7.0 or later |
| **Java** | 17 or later |

!!! warning "Spring Boot 4 required as of inference4j 0.11.0"

    The starter targets Spring Boot 4. It will not load in a Spring Boot 3
    application — Boot 4 relocated the actuator health types, so the two lines are
    binary-incompatible and cannot be served by one artifact.

    If you are still on Spring Boot 3, stay on
    **`inference4j-spring-boot-starter:0.10.1`**, which is the last release built
    against Boot 3.4. It is frozen, not maintained — Spring Boot 3.x itself left
    open-source support on 30 June 2026, so migrating to Boot 4 is the recommended
    path. See [Migrating from 0.10.x](#migrating-from-010x) below.

    Only the starter is affected. `inference4j-core` and every other module are
    plain Java libraries with no Spring dependency, and work unchanged on any
    Spring version or none at all.

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
| `inference4j.actuator.models.enabled` | `boolean` | `true` | Create the [`models` actuator endpoint](#model-cache-endpoint) bean when Spring Boot Actuator is on the classpath. It still has to be exposed with `management.endpoints.web.exposure.include` |
| `inference4j.dashboard.enabled` | `boolean` | `false` | Serve the [dashboard](#dashboard) in a Spring MVC web application |
| `inference4j.dashboard.path` | `String` | `/inference4j` | The path the dashboard is served at; its JSON API is under `<path>/api` |
| `inference4j.dashboard.read-only` | `boolean` | `true` | When `false`, the dashboard can update, evict and prune cached models |
| `inference4j.health.model-cache-ttl` | `Duration` | `10s` | How long the health indicator reuses its listing of the model cache (`staleModels`, `updatesAvailable`), so frequent probes don't read the cache each time. `0s` lists it on every call |
| `inference4j.metrics.enabled` | `boolean` | `true` | Use a Micrometer-backed `RouterMetrics` when a `MeterRegistry` is present. When `false`, or when Micrometer is absent, `RouterMetrics` falls back to a no-op instead of being removed |
| `inference4j.nlp.text-classifier.enabled` | `boolean` | `false` | Enable DistilBERT text classifier |
| `inference4j.nlp.text-classifier.model-id` | `String` | `inference4j/distilbert-base-uncased-finetuned-sst-2-english` | Model ID |
| `inference4j.nlp.text-embedder.enabled` | `boolean` | `false` | Enable SentenceTransformer embedder |
| `inference4j.nlp.text-embedder.model-id` | `String` | *(required)* | Model ID (no default — must be specified) |
| `inference4j.nlp.search-reranker.enabled` | `boolean` | `false` | Enable MiniLM cross-encoder reranker |
| `inference4j.nlp.search-reranker.model-id` | `String` | `inference4j/ms-marco-MiniLM-L-6-v2` | Model ID |
| `inference4j.nlp.named-entity-recognizer.enabled` | `boolean` | `false` | Enable BERT NER recognizer |
| `inference4j.nlp.named-entity-recognizer.model-id` | `String` | `inference4j/distilbert-NER` | Model ID |
| `inference4j.vision.image-classifier.enabled` | `boolean` | `false` | Enable ResNet image classifier |
| `inference4j.vision.image-classifier.model-id` | `String` | `inference4j/resnet50-v1-7` | Model ID |
| `inference4j.vision.object-detector.enabled` | `boolean` | `false` | Enable the object detector |
| `inference4j.vision.object-detector.architecture` | `String` | `detr` | Wrapper that builds the bean: `detr` (`DetrDetector`), `yolov8` (`YoloV8Detector`) or `yolo26` (`Yolo26Detector`). See [Object detection](#object-detection) |
| `inference4j.vision.object-detector.model-id` | `String` | the architecture's default | Model ID. Unset, it is `inference4j/dfine-small-coco` for `detr`, `inference4j/yolov8n` for `yolov8` and `inference4j/yolo26n` for `yolo26` |
| `inference4j.vision.text-detector.enabled` | `boolean` | `false` | Enable CRAFT text detector |
| `inference4j.vision.text-detector.model-id` | `String` | `inference4j/craft-mlt-25k` | Model ID |
| `inference4j.audio.speech-recognizer.enabled` | `boolean` | `false` | Enable Wav2Vec2 speech recognizer |
| `inference4j.audio.speech-recognizer.model-id` | `String` | `inference4j/wav2vec2-base-960h` | Model ID |
| `inference4j.audio.vad.enabled` | `boolean` | `false` | Enable Silero VAD |
| `inference4j.audio.vad.model-id` | `String` | `inference4j/silero-vad` | Model ID |
| `inference4j.genai.text-generator.enabled` | `boolean` | `false` | Enable the onnxruntime-genai `TextGenerator` (requires `inference4j-genai`) |
| `inference4j.genai.text-generator.model` | `String` | `phi-3-mini` | Preset: `phi-3-mini`, `phi-4-mini`, `deepseek-r1-1.5b`, `qwen3-0.6b` or `smollm3-3b` |
| `inference4j.genai.text-generator.max-length` / `temperature` / `top-k` / `top-p` | number | builder defaults | Generation settings; unset keeps the builder's default |
| `inference4j.genai.vision-language.enabled` | `boolean` | `false` | Enable `VisionLanguageModel` (requires `inference4j-genai`) |
| `inference4j.genai.vision-language.model` | `String` | `phi-3.5-vision` | Preset: `phi-3.5-vision` |
| `inference4j.genai.vision-language.max-length` / `temperature` / `top-k` / `top-p` | number | builder defaults | Generation settings |
| `inference4j.genai.whisper.enabled` | `boolean` | `false` | Enable `WhisperSpeechModel`; it also becomes the `SpeechRecognizer` bean (requires `inference4j-genai`) |
| `inference4j.genai.whisper.model-id` | `String` | `inference4j/whisper-small-genai` | Model ID (`inference4j/whisper-tiny-genai` for the small, fast one) |
| `inference4j.genai.whisper.language` | `String` | `en` | Source language code |
| `inference4j.genai.whisper.task` | `String` | `transcribe` | `transcribe` or `translate` (to English) |
| `inference4j.genai.whisper.max-length` | `int` | `448` | Maximum tokens per 30-second chunk |

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

### Object detection

`inference4j.vision.object-detector.enabled=true` registers an `ObjectDetector` built with
`DetrDetector` and D-FINE small (`inference4j/dfine-small-coco`, Apache-2.0 weights). Its default
confidence threshold is 0.5.

To use a YOLO model, set `architecture`. Without a `model-id`, each architecture loads its own
default model, so the D-FINE weights are never handed to a YOLO wrapper:

```yaml
inference4j:
  vision:
    object-detector:
      enabled: true
      architecture: yolov8          # loads inference4j/yolov8n (AGPL-3.0 weights)
```

A `model-id` must match the `architecture`: other DETR-family models such as
`inference4j/dfine-nano-coco` or `inference4j/rtdetr-v2-r18vd` work with `detr`, and a YOLOv8
checkpoint needs `architecture: yolov8`.

!!! warning "Changed in 0.15.0"
    Before 0.15.0 the bean was always `YoloV8Detector` with `inference4j/yolov8n` and a confidence
    threshold of 0.25. Applications that set a YOLO `model-id` must now also set `architecture`,
    or the model is loaded by `DetrDetector`.

### Generative models

Add `inference4j-genai` next to the starter, and the text generator, vision-language model and
Whisper become available with the same opt-in properties:

```yaml
inference4j:
  genai:
    text-generator:
      enabled: true
      model: phi-3-mini
      max-length: 512
    whisper:
      enabled: true
      model-id: inference4j/whisper-tiny-genai
```

```java
public AssistantController(TextGenerator generator, SpeechRecognizer recognizer) { … }
```

Without `inference4j-genai` on the classpath, the `inference4j.genai.*` properties are ignored.

`WhisperSpeechModel` is a `SpeechRecognizer`, so with `inference4j.genai.whisper.enabled=true` it
**replaces** Wav2Vec2 as the application's `SpeechRecognizer` bean, even if
`inference4j.audio.speech-recognizer.enabled` is also set. Inject `WhisperSpeechModel` directly for
Whisper-only options. For recordings longer than 30 seconds, wrap it yourself in a
[`SegmentedSpeechRecognizer`](../use-cases/speech-to-text.md#long-recordings) bean.

Presets cover the models with known chat templates. For any other genai model, define your own
`TextGenerator` bean with `modelSource` and `chatTemplate`; it replaces the auto-configured one.

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

An actuator health indicator is included automatically when Spring Boot Actuator is on the classpath. It reports the number of registered `InferenceTask` and `GenerativeTask` beans (a bean that is both, like `WhisperSpeechModel`, counts once).

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
        "tasks": 3,
        "staleModels": [
          {
            "repo": "inference4j/silero-vad",
            "loadedRevisions": ["3f1c..."],
            "cachedRevision": "9a7e..."
          }
        ],
        "updatesAvailable": ["inference4j/distilbert-NER"]
      }
    }
  }
}
```

The two model cache details come from the [model cache](model-loading.md#managing-the-cache) on disk, without
calling the Hub:

| Detail | Description |
|--------|-------------|
| `staleModels` | Repos where a model loaded in this JVM reads an older snapshot whose files differ from the cached revision, so it serves an older version than a model loaded now would. `loadedRevisions` are the snapshots still in use. |
| `updatesAvailable` | Repos whose last Hub check found changed files that are not downloaded yet. |

Neither changes the status: a stale model still serves requests. Restarting the application, or rebuilding the
model bean, loads the cached revision. The details appear when a `ModelCache` bean exists, which the starter
defines by default.

The health indicator is enabled by default. Disable it with:

```yaml
inference4j:
  health:
    enabled: false
```

## Model cache endpoint

With Spring Boot Actuator on the classpath, the starter adds a `models` endpoint over the
[model cache](model-loading.md#managing-the-cache): it lists the cached models, checks them for updates, and can
update, evict and prune them. Like most actuator endpoints, it is not exposed over HTTP until you include it:

```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,models
```

### Operations

| Request | Access | Description | Status |
|---------|--------|-------------|--------|
| `GET /actuator/models` | read | Every cached repo: revision, last check, snapshots, files, leases, size, `stale`, `updating`. Local state only. | 200 |
| `GET /actuator/models/{org}/{name}` | read | One cached repo. | 200, 404 when not cached |
| `GET /actuator/models/{org}/{name}?check=true` | read | Also asks the Hub which cached files changed, without downloading. A check younger than a minute is reused. | 200, 404, 409 offline, 502 Hub unreachable |
| `POST /actuator/models/{org}/{name}` with `{"action":"update"}` | write | Starts an update in the background: downloads the changed files into a new snapshot. | 202, 400 unknown action, 404 |
| `DELETE /actuator/models/{org}/{name}` | write | Evicts the repo: deletes its snapshots that are not in use. | 200, 404 |
| `DELETE /actuator/models/{org}/{name}?revision={commit}` | write | Deletes one snapshot, unless it is in use. | 200, 404 |
| `DELETE /actuator/models` | write | Prunes every repo: deletes superseded snapshots that are not in use, and unfinished downloads. | 200 |

An invalid repo id or revision answers 400. Errors have a JSON body with `status` and `message`. Responses list
paths inside the repo only, never the cache directory.

```bash
# every cached model
curl http://localhost:8080/actuator/models

# one model, asking the Hub whether it changed
curl 'http://localhost:8080/actuator/models/inference4j/silero-vad?check=true'
```

```json
{
  "repoId": "inference4j/silero-vad",
  "revision": "9a7e...",
  "lastChecked": "2026-11-22T10:00:00Z",
  "remoteRevision": "b21d...",
  "updateAvailable": true,
  "stale": false,
  "updating": false,
  "snapshots": [
    {
      "revision": "9a7e...",
      "current": true,
      "leases": 1,
      "files": [{ "path": "model.onnx", "id": "4c5f...", "sizeBytes": 2313101 }],
      "sizeBytes": 2313101
    }
  ],
  "sizeBytes": 2313101,
  "check": {
    "cachedRevision": "9a7e...",
    "remoteRevision": "b21d...",
    "changedFiles": ["model.onnx"],
    "checkedAt": "2026-11-22T10:00:00Z",
    "updateAvailable": true
  }
}
```

The write operations, once [enabled](#enabling-writes):

```bash
# update in the background; GET the model to follow it ("updating", then the new "revision")
curl -X POST http://localhost:8080/actuator/models/inference4j/silero-vad \
     -H 'Content-Type: application/json' -d '{"action":"update"}'

# evict a model, or one of its snapshots
curl -X DELETE http://localhost:8080/actuator/models/inference4j/silero-vad
curl -X DELETE 'http://localhost:8080/actuator/models/inference4j/silero-vad?revision=9a7e...'

# prune every repo
curl -X DELETE http://localhost:8080/actuator/models
```

Updates run one at a time, on a thread the endpoint owns. A request for a repo whose update is already running
answers 202 with `"status": "already-running"`. A failed update is logged as a warning, and the repo keeps its
current snapshot. Models already loaded keep reading their snapshot after an update; models loaded afterwards read
the new one. An eviction keeps the snapshots in use, listed under `inUse` in the response.

### Enabling writes

The endpoint's default access is read-only: `POST` and `DELETE` are not mapped, and answer 405. To allow them:

```yaml
management:
  endpoint:
    models:
      access: unrestricted
```

`management.endpoints.access.max-permitted` caps this setting, as for every endpoint.

### Path

The URL follows the actuator base path and can be moved like any endpoint's:

```yaml
management:
  endpoints:
    web:
      path-mapping:
        models: model-cache      # /actuator/model-cache
```

### Disabling the endpoint

Leaving `models` out of the exposure list keeps it off HTTP. To not create the bean at all, for example to define
your own endpoint with the id `models` over the `ModelCache` bean:

```yaml
inference4j:
  actuator:
    models:
      enabled: false
```

`management.endpoint.models.access: none` also removes it.

### Which cache

The endpoint and the health details use the `ModelCache` bean. By default it is the cache of your
`HuggingFaceModelSource` bean when you define one, and otherwise `ModelCache.defaultCache()`, the cache the
auto-configured models download into. Define a `ModelCache` bean to use another one.

### Security

- The endpoint is not exposed over HTTP by default.
- `?check=true` makes an outbound request to the Hub (or the mirror in `HF_ENDPOINT`). It runs only for repos
  already cached, so a request cannot make the application contact the Hub about an arbitrary repo id.
- The write operations can fill the disk (`update` downloads files) or force the next model loads to download
  again (`DELETE`). When you enable them, put the endpoint behind authentication, and keep CSRF protection on for
  `POST` and `DELETE`, for example with Spring Security's `EndpointRequest.to("models")` matcher.
- `update` pulls the latest commit of the repo's `main` branch. It does not pin a revision.

## Dashboard

The starter can serve a local dashboard, a web page over the [model cache](model-loading.md#managing-the-cache).
It shows the cache's size and the disk usage of its volume, the largest models, and every cached repo with its
snapshots and files. When writes are enabled, it can also check repos for updates, update, evict and prune them.

The dashboard needs a Spring MVC web application (`spring-boot-starter-webmvc`) and is off by default:

```yaml
inference4j:
  dashboard:
    enabled: true
    read-only: false     # allow update, evict and prune; checking for updates is always allowed
```

It is served at `http://localhost:8080/inference4j/`, and the URL is logged at startup. Its page, styles and
scripts ship in `inference4j-core`; there are no requests to a CDN, so it also works offline. It uses the same
`ModelCacheService` bean as the [model cache endpoint](#model-cache-endpoint), so updates started from either one
show as `updating` in both.

The page reads a small JSON API, which other tools can call too:

| Request | Description | Status |
|---------|-------------|--------|
| `GET /inference4j/api/info` | `{"readOnly": true}` | 200 |
| `GET /inference4j/api/cache` | The cache summary: models, snapshots, sizes, and the volume's `disk.totalBytes` and `disk.usableBytes` | 200 |
| `GET /inference4j/api/cache/models` | Every cached repo | 200 |
| `GET /inference4j/api/cache/model?repo={org}/{name}[&check=true]` | One cached repo; with `check`, after asking the Hub | 200, 400, 404, 409 offline, 502 Hub unreachable |
| `POST /inference4j/api/cache/update?repo={org}/{name}` | Starts an update in the background | 202, 403 read-only, 404 |
| `DELETE /inference4j/api/cache/model?repo={org}/{name}[&revision={commit}]` | Evicts the repo, or one snapshot | 200, 403, 404 |
| `POST /inference4j/api/cache/prune` | Prunes superseded snapshots not in use | 200, 403 |

### Dashboard security

- The dashboard has no authentication of its own. It is meant for local development; in a deployed application,
  leave it off or put its path behind your Spring Security configuration.
- Like the endpoint's `?check=true`, a check runs only for repos already cached.
- With Spring Security's cookie CSRF repository (`CookieCsrfTokenRepository`), the page sends the `XSRF-TOKEN`
  cookie back as the `X-XSRF-TOKEN` header on `POST` and `DELETE`.

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

## Migrating from 0.10.x

inference4j 0.11.0 moved the starter from Spring Boot 3 to Spring Boot 4. There is no
Boot 3 branch — 0.10.1 is the last Boot 3 release and stays available on Maven Central.

**1. Upgrade your application to Spring Boot 4.** Spring's own
[migration guide](https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-4.0-Migration-Guide)
recommends going to 3.5 first, then to 4.0. Boot 4 requires Java 17 or later.

**2. Update the health indicator import, if you override it.** The actuator health types
moved modules in Boot 4. If you replace inference4j's default indicator with your own bean,
change:

```java
// Before (Spring Boot 3)
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;

// After (Spring Boot 4)
import org.springframework.boot.health.contributor.Health;
import org.springframework.boot.health.contributor.HealthIndicator;
```

The interface itself is unchanged — `HealthIndicator` is still a `@FunctionalInterface`
returning `Health`, so an existing lambda keeps working once the import is fixed.

**3. Nothing else changes.** Every `inference4j.*` property, bean name and default is
identical to 0.10.x. If you do not override the health indicator, upgrading your Boot
version is the only step.

### If you cannot move to Spring Boot 4 yet

Pin the starter to the last Boot 3 release and let it float independently of the rest:

=== "Gradle"

    ```groovy
    implementation 'io.github.inference4j:inference4j-spring-boot-starter:0.10.1'
    implementation 'io.github.inference4j:inference4j-core:${inference4jVersion}'
    ```

=== "Maven"

    ```xml
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-spring-boot-starter</artifactId>
        <version>0.10.1</version>
    </dependency>
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-core</artifactId>
        <version>${inference4jVersion}</version>
    </dependency>
    ```

`inference4j-core` has no Spring dependency, so you can keep it current while the starter
stays pinned. The 0.10.1 starter will not receive fixes.

## Tips

- Use `model-id` to swap models without changing code (e.g., switch from ResNet to EfficientNet).
- The text embedder has no default model ID — you must specify one via `model-id`.
- All beans implement `AutoCloseable` and are properly cleaned up on application shutdown.
