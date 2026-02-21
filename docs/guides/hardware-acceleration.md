# Hardware Acceleration

inference4j supports GPU and hardware acceleration via ONNX Runtime execution providers. The `.sessionOptions()` API is available on every model wrapper.

## CoreML (macOS)

CoreML is bundled in the standard ONNX Runtime dependency on macOS — no additional setup needed.

```java
try (var classifier = ResNetClassifier.builder()
        .sessionOptions(opts -> opts.addCoreML())
        .build()) {
    classifier.classify(Path.of("cat.jpg"));
}
```

## CUDA (Linux/Windows)

For NVIDIA GPU acceleration, swap the ONNX Runtime dependency:

=== "Gradle"

    ```groovy
    implementation('io.github.inference4j:inference4j-core:${inference4jVersion}') {
        exclude group: 'com.microsoft.onnxruntime', module: 'onnxruntime'
    }
    implementation 'com.microsoft.onnxruntime:onnxruntime_gpu:${onnxruntimeVersion}'
    ```

=== "Maven"

    ```xml
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-core</artifactId>
        <version>${inference4jVersion}</version>
        <exclusions>
            <exclusion>
                <groupId>com.microsoft.onnxruntime</groupId>
                <artifactId>onnxruntime</artifactId>
            </exclusion>
        </exclusions>
    </dependency>
    <dependency>
        <groupId>com.microsoft.onnxruntime</groupId>
        <artifactId>onnxruntime_gpu</artifactId>
        <version>${onnxruntimeVersion}</version>
    </dependency>
    ```

Then enable CUDA in the builder:

```java
try (var classifier = ResNetClassifier.builder()
        .sessionOptions(opts -> opts.addCUDA(0))  // device ID 0
        .build()) {
    classifier.classify(Path.of("cat.jpg"));
}
```

## The `sessionOptions` API

Every model wrapper exposes `.sessionOptions(SessionConfigurer)` in its builder. `SessionConfigurer` is a `@FunctionalInterface` that receives the ONNX Runtime `SessionOptions`:

```java
@FunctionalInterface
public interface SessionConfigurer {
    void configure(OrtSession.SessionOptions options) throws OrtException;
}
```

This gives you full access to ONNX Runtime configuration:

```java
.sessionOptions(opts -> {
    opts.addCoreML();
    opts.setIntraOpNumThreads(4);
})
```

### Common options

| Method | Description |
|--------|-------------|
| `opts.addCoreML()` | Enable CoreML (macOS) |
| `opts.addCUDA(deviceId)` | Enable CUDA (Linux/Windows) |
| `opts.setIntraOpNumThreads(n)` | Set number of threads for intra-op parallelism |
| `opts.setInterOpNumThreads(n)` | Set number of threads for inter-op parallelism |
| `opts.setOptimizationLevel(level)` | Set graph optimization level |

## Benchmarks on Apple Silicon (M-series)

| Model | Capability | CPU | CoreML | Speedup |
|-------|------------|-----|--------|---------|
| ResNet-50 | Image Classification | 37 ms | 10 ms | **3.7x** |
| CRAFT | Text Detection | 831 ms | 153 ms | **5.4x** |

Measured with 3 warmup runs + 10 timed runs.

## Tips

- CoreML is available on macOS only. The `addCoreML()` call will fail on other platforms.
- CUDA requires the `onnxruntime_gpu` artifact and a compatible NVIDIA driver + CUDA toolkit.
- If the execution provider fails to initialize, ONNX Runtime silently falls back to CPU. Check logs for warnings.
- For production workloads, benchmark both CPU and GPU — small models (like MiniLM) may be faster on CPU due to GPU data transfer overhead.
- `.sessionOptions()` is composable — you can set multiple options in a single lambda.
