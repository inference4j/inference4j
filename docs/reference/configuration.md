# Configuration

## Model cache

Models are downloaded from HuggingFace and cached locally. The cache directory is resolved in this order:

| Priority | Method | Example |
|----------|--------|---------|
| 1 | Constructor parameter | `new HuggingFaceModelSource(Path.of("/cache"))` |
| 2 | System property | `-Dinference4j.cache.dir=/path/to/cache` |
| 3 | Environment variable | `INFERENCE4J_CACHE_DIR=/path/to/cache` |
| 4 | Default | `~/.cache/inference4j/` |

## JVM flags

ONNX Runtime requires native access:

```
--enable-native-access=ALL-UNNAMED
```

Or, on the module path:

```
--enable-native-access=com.microsoft.onnxruntime
```

## System properties

| Property | Description | Default |
|----------|-------------|---------|
| `inference4j.cache.dir` | Model cache directory | `~/.cache/inference4j/` |

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE4J_CACHE_DIR` | Model cache directory | `~/.cache/inference4j/` |

## Spring Boot properties

See the [Spring Boot guide](../guides/spring-boot.md#all-properties) for the full list of `inference4j.*` application properties.

## ONNX Runtime session options

Session-level configuration is set via `.sessionOptions()` on each builder:

```java
.sessionOptions(opts -> {
    opts.addCoreML();                                      // execution provider
    opts.setIntraOpNumThreads(4);                          // parallelism
    opts.setOptimizationLevel(SessionOptions.OptLevel.ALL_OPT); // graph optimization
})
```

See the [Hardware Acceleration guide](../guides/hardware-acceleration.md) for execution provider details.
