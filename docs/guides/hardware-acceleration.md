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

## Plugin execution providers

ONNX Runtime can also load execution providers from **separately packaged plugin libraries**, rather than only the ones compiled into the runtime. Microsoft ships several this way — the CUDA and WebGPU plugin EPs, for example — versioned independently of ONNX Runtime itself.

This is additive. `addCoreML()` and `addCUDA()` above are unaffected and remain the simplest path. Reach for a plugin EP when you need a backend that isn't built into your ONNX Runtime distribution.

### Register the plugin library

Registration happens on the `OrtEnvironment`, not on the session. inference4j uses ONNX Runtime's default environment, so anything you register there applies to every model you build afterwards:

```java
OrtEnvironment env = OrtEnvironment.getEnvironment();
env.registerExecutionProviderLibrary("my_ep", "/path/to/plugin_ep_library.so");
```

### Pick devices and attach them

`getEpDevices()` lists every execution provider / device combination available, including ones contributed by registered plugins. Filter down to the provider you want, then pass the result to `addExecutionProvider`:

```java
OrtEnvironment env = OrtEnvironment.getEnvironment();

List<OrtEpDevice> devices = env.getEpDevices().stream()
        .filter(d -> "my_ep".equals(d.getEpName()))
        .toList();

try (var classifier = ResNetClassifier.builder()
        .sessionOptions(opts -> opts.addExecutionProvider(devices, Map.of()))
        .build()) {
    classifier.classify(Path.of("cat.jpg"));
}
```

Two rules worth knowing:

- Every `OrtEpDevice` in the list must belong to the **same** execution provider, though they may point at different devices.
- Providers apply **in the order added** — the first provider added gets first choice when graph nodes are assigned.

### Inspect what's available

`OrtEpDevice` exposes `getEpName()`, `getEpVendor()`, `getEpMetadata()`, `getEpOptions()` and `getDevice()`, which is enough to see what a machine actually offers:

```java
for (OrtEpDevice d : OrtEnvironment.getEnvironment().getEpDevices()) {
    System.out.println(d.getEpName() + " (" + d.getEpVendor() + ") -> " + d.getDevice());
}
```

To remove a plugin again:

```java
env.unregisterExecutionProviderLibrary("my_ep");
```

!!! note "Requires ONNX Runtime 1.24.3 or later"

    `registerExecutionProviderLibrary()`, `getEpDevices()` and `addExecutionProvider()` are not
    present in 1.23. inference4j depends on ONNX Runtime 1.26.0 as of 0.10.1, so they are
    available with no extra setup.

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
| `opts.addExecutionProvider(devices, options)` | Enable a [plugin execution provider](#plugin-execution-providers) |
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
