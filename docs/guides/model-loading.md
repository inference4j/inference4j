# Model Loading

inference4j resolves models through the `ModelSource` interface. By default, models are downloaded from HuggingFace and cached locally. You can also load models from disk or implement your own resolution strategy.

## ModelSource interface

```java
@FunctionalInterface
public interface ModelSource {
    Path resolve(String modelId);

    // Defaults, overridden by HuggingFaceModelSource
    default ResolvedModel open(String modelId) { ... }
    default ResolvedModel open(String modelId, List<String> requiredFiles) { ... }
    default ResolvedModel open(String modelId, String subdirectory) { ... }
}
```

`ModelSource` takes a model ID and returns the local directory path where the model files live. Every builder accepts `.modelSource(ModelSource)`.

Builders call `open(...)`, which returns a `ResolvedModel`: the directory, plus a handle that keeps it in use until it
is closed. For `HuggingFaceModelSource`, an open handle keeps the snapshot it points to on disk, so updates and
pruning never delete files a model is loading or using. Lambdas and `LocalModelSource` return unmanaged handles,
whose `close()` does nothing.

```java
try (ResolvedModel model = source.open("inference4j/silero-vad", List.of("model.onnx"))) {
    InferenceSession session = InferenceSession.create(model, "model.onnx");
    // the session retains the snapshot until session.close()
}
```

## LocalModelSource

Load models from a directory on disk:

```java
var source = new LocalModelSource(Path.of("/models"));

try (var classifier = DistilBertTextClassifier.builder()
        .modelId("my-sentiment-model")
        .modelSource(source)
        .build()) {
    classifier.classify("Great product!");
}
```

This resolves to `/models/my-sentiment-model/`, which should contain the model files (`model.onnx`, `vocab.txt`, `config.json`, etc.).

## HuggingFaceModelSource

The default model source. Downloads models from the [HuggingFace Hub](https://huggingface.co/) and caches them locally.

```java
// Uses default cache directory (~/.cache/inference4j/)
var source = HuggingFaceModelSource.defaultInstance();

// Or specify a custom cache directory
var source = new HuggingFaceModelSource(Path.of("/custom/cache"));

// Or set every option
var source = HuggingFaceModelSource.builder()
        .cacheDir(Path.of("/custom/cache"))
        .checkInterval(Duration.ofHours(1))   // default: a day
        .offline(false)
        .endpoint(URI.create("https://hf-mirror.example.com"))
        .pruneMode(PruneMode.AUTO)                     // default; MANUAL keeps old snapshots
        .minimumSnapshotAge(Duration.ofMinutes(10))  // default
        .build();
```

### Cache layout

Each repo is cached as one or more snapshots, one per Hub commit:

```
~/.cache/inference4j/
  hub/
    .locks/models--inference4j--silero-vad.lock
    models--inference4j--silero-vad/
      refs/main                   the current snapshot's commit
      snapshots/{commit}/         the files at that commit
      manifests/{commit}.json     each file's Hub id and size
      repo.json                   the last update check
```

An update writes a new snapshot next to the current one and switches `refs/main` to it once every file is in place,
so a model never reads files from two revisions. Files that did not change are reused, never downloaded again:

- **Moved** from the old snapshot when no model in this JVM uses it and it is older than the
  [minimum age](#pruning-and-snapshots-in-use). The update then only takes the space of the files that changed. The
  old snapshot keeps only its versions of the changed files, is never served again, and is pruned like any
  superseded snapshot.
- **Copied** when a model in this JVM still uses the old snapshot, or the snapshot is younger than the minimum age,
  so that the running model keeps its files. An update while the old revision is in use needs disk space for both
  copies of the unchanged files until the old snapshot is pruned.

The cache never hard-links model files: ONNX Runtime refuses external data (`model.onnx.data`) that has more than one
hard link. A snapshot file that has other hard links, as caches written by development builds before 0.15.0 have, is
replaced by a private copy when it is opened, and a link back to the flat layout is removed instead (logged once per
repo at INFO). Where the link count cannot be read, as on Windows, files are left as they are; a snapshot that a
loaded model uses is left as it is too, with a warning.

`resolve(...)` returns the current snapshot's directory, such as
`~/.cache/inference4j/hub/models--inference4j--silero-vad/snapshots/{commit}/`. Code that built paths as
`{cacheDir}/{org}/{name}/` should use the returned path instead. A path from `resolve` is not protected from pruning;
`open` returns a handle that is.

### Updates

Cached models stay in sync with the Hub. At most once per check interval (a day by default), opening a cached model
asks the Hub for the repo's current revision, with one small API call.

- **Changed files** are downloaded at the new commit into a new snapshot; unchanged files are moved or copied into
  it. A model card
  edit changes no cached file, so it creates no snapshot and downloads nothing; the check is recorded in `repo.json`.
- **Files requested between checks**, such as a file a second wrapper needs from the same repo, are downloaded at the
  commit of the current snapshot, never at the Hub's latest.
- **Models already loaded** keep reading the snapshot they were loaded from. Models loaded afterwards read the new
  one.
- **Hub unreachable:** if the check fails (no network, timeout, an error response), the cached files are used and a
  warning is logged once. If a download fails during an update, the current snapshot stays in use, a warning is
  logged, and the next check resumes the update without downloading again the files already fetched.
- **Revision API unavailable but downloads working:** missing files are downloaded, and the snapshot is named after
  the commit the Hub reports with the download (`X-Repo-Commit`), or `unknown` without it.
- **Offline mode** never contacts the Hub: cached files are used as they are, and a model that is not fully
  cached fails to load. Turn it on with `INFERENCE4J_OFFLINE=true`, `-Dinference4j.offline=true`,
  `HF_HUB_OFFLINE=1` or `.offline(true)`.
- **Check interval:** `INFERENCE4J_CACHE_CHECK_INTERVAL` or `-Dinference4j.cache.check-interval` take an
  ISO-8601 duration, such as `PT1H` or `P7D`; `PT0S` checks on every load.
- **Mirrors:** `HF_ENDPOINT` or `-Dinference4j.hf.endpoint` point to a Hub mirror.

### Caches from earlier versions

Versions before 0.15.0 cached files flat, in `{cacheDir}/{org}/{name}/`. These caches are adopted the first time
each repo is opened, without downloading again:

- **Online:** each cached file is hashed once and compared with the Hub's id. Files that match are moved into the
  new snapshot; files that differ are downloaded.
- **Offline:** the files are moved into a snapshot named `unknown`, and checked against the Hub at the next online
  open, which promotes them to a snapshot named after the Hub's commit.
- Adopted files leave the flat layout, so they take no extra disk space. An older inference4j version sharing the
  cache no longer finds them there and downloads them again. Where a file cannot be moved (another file system), it
  is copied and the flat file deleted. `ModelCache.prune()` deletes flat files left at the paths of adopted ones.
- **Read-only cache:** when the cache directory cannot be written, cached files (flat or snapshot) are read in place,
  without update checks or eviction protection, and a warning is logged once.

### Managing the cache

`HuggingFaceModelSource.cache()` returns a `ModelCache`, and `ModelCache.defaultCache()` is the default source's:

```java
ModelCache cache = ModelCache.defaultCache();

for (CachedModel model : cache.list()) {          // local state only
    System.out.printf("%s %s %d bytes%n", model.repoId(), model.revision(), model.sizeBytes());
}

UpdateCheck check = cache.check("inference4j/silero-vad");   // asks the Hub, downloads nothing
if (check.updateAvailable()) {
    cache.update("inference4j/silero-vad");                  // downloads the changed files now
}

cache.evict("inference4j/silero-vad");            // deletes the repo's snapshots not in use
cache.prune();                                    // deletes superseded snapshots of every repo
```

| Method | Description |
|--------|-------------|
| `list()` | Every cached repo: current revision, last check, remote revision, snapshots, files, leases and size. Reads local state only. |
| `find(repoId)` | One cached repo, or empty. |
| `check(repoId)` | Asks the Hub which cached files changed, without downloading. The result is recorded, so the next load fetches them. |
| `update(repoId)` | Brings the repo up to date now, whatever the check interval. |
| `evict(repoId)` | Deletes the repo's snapshots that are not in use, and its adopted flat files. |
| `evict(repoId, revision)` | Deletes one snapshot, unless it is in use. |
| `prune()` | Deletes superseded snapshots that are not in use, unfinished downloads, and adopted flat files, in every repo. |

`CachedModel.stale()` is true when a model still loaded in this JVM reads an older snapshot whose files differ from
the current one: it serves an older version of the model than a model loaded now would. Evictions return an
`EvictionResult` with the snapshots removed, those kept because they are in use, and the bytes freed (a file still
hard-linked from a kept snapshot, as by development builds before 0.15.0, does not count).

In a Spring Boot application, the starter's [`models` actuator endpoint](spring-boot.md#model-cache-endpoint)
offers these operations over HTTP, and its health indicator reports stale models and available updates.

### Pruning and snapshots in use

A snapshot is in use while a handle on it is open: a builder holds one while it loads, and each `InferenceSession`
(and each genai model) created from it holds its own until it is closed. Pruning and eviction skip snapshots in use.

- **Pruning mode:** with `auto` (the default), the snapshots an update supersedes are deleted right after it, unless
  they are in use. With `manual`, they stay until `prune()` or `evict(...)`. Set it with
  `-Dinference4j.cache.prune=manual`, `INFERENCE4J_CACHE_PRUNE=manual` or `.pruneMode(PruneMode.MANUAL)`.
- **Minimum age:** leases are counted in the JVM only, so a snapshot another process sharing the cache is still
  loading cannot be seen as in use. Pruning and eviction therefore never delete a snapshot younger than the minimum
  age, 10 minutes by default, counted from its creation. Set it with `.minimumSnapshotAge(Duration)`.
- **Other processes:** a process that loaded an older snapshot and keeps running is not protected once that
  snapshot is past the minimum age: an update may move its unchanged files into the new snapshot, and pruning may
  delete it. On Linux and macOS, files moved or deleted while open (such as memory-mapped external
  weights in `model.onnx.data`) stay readable until closed; single-file models are read into memory when the session
  is built. On Windows, a snapshot with open files cannot be deleted; it is reported as in use and left for a later
  prune.
- **Locking:** updates, adoption and eviction of a repo run under an exclusive lock file in `hub/.locks/`, so two
  processes never download the same repo at once; one waits for the other. File locks are not reliable on network
  file systems (NFS, SMB); give each host its own cache directory there.

### Cache directory resolution

The cache directory is resolved in this order:

1. Constructor parameter (`new HuggingFaceModelSource(path)`)
2. System property: `-Dinference4j.cache.dir=/path/to/cache`
3. Environment variable: `INFERENCE4J_CACHE_DIR=/path/to/cache`
4. Default: `~/.cache/inference4j/`

### Downloaded files

`HuggingFaceModelSource` attempts to download these files (skipping any that don't exist):

- `model.onnx` — the ONNX model
- `vocab.txt` — WordPiece vocabulary (NLP models)
- `vocab.json` — JSON vocabulary (audio models)
- `config.json` — HuggingFace model config
- `labels.txt` — class labels (vision models)

## Lambda shorthand

Since `ModelSource` is a `@FunctionalInterface`, you can use a lambda:

```java
try (var classifier = ResNetClassifier.builder()
        .modelId("resnet50")
        .modelSource(id -> Path.of("/models").resolve(id))
        .build()) {
    classifier.classify(Path.of("cat.jpg"));
}
```

## Required model files by task

| Task | Required files |
|------|----------------|
| Text classification | `model.onnx`, `vocab.txt`, `config.json` (with `id2label`) |
| Text embeddings | `model.onnx`, `vocab.txt` |
| Search reranking | `model.onnx`, `vocab.txt` |
| Image classification | `model.onnx`, `labels.txt` |
| Object detection | `model.onnx`, `labels.txt` |
| Text detection | `model.onnx` |
| Speech-to-text | `model.onnx`, `vocab.json` |
| Voice activity detection | `model.onnx` |

## Using a custom HuggingFace model

Any ONNX-exported model hosted on HuggingFace works — just override the model ID:

```java
try (var classifier = DistilBertTextClassifier.builder()
        .modelId("your-org/your-fine-tuned-model")
        .build()) {
    classifier.classify("Some text");
}
```

The model repository must contain the required files listed above.

## Tips

- `LocalModelSource` does no downloading — it expects files to already exist at the resolved path.
- `HuggingFaceModelSource` is thread-safe. Concurrent requests for the same model wait for the first download to complete.
  Sources on the same cache directory share their locks and leases.
- Cached models are checked for updates once a day (see [Updates](#updates)). To force a full re-download,
  call `cache().evict(repoId)` once no model uses the repo.
- The model ID is just a string — it can be any identifier meaningful to your `ModelSource` implementation.
