# Installation

## Requirements

- **Java 17** or higher
- ONNX Runtime 1.26.0 (included transitively via `inference4j-core`)

## Add the dependency

`inference4j-core` is the only dependency you need — it includes all task wrappers, preprocessing, and tokenizers.

=== "Gradle"

    ```groovy
    implementation 'io.github.inference4j:inference4j-core:${inference4jVersion}'
    ```

=== "Maven"

    ```xml
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-core</artifactId>
        <version>${inference4jVersion}</version>
    </dependency>
    ```

## Generative AI

For text generation (Phi-3, DeepSeek-R1, etc.), add `inference4j-genai` instead:

=== "Gradle"

    ```groovy
    implementation 'io.github.inference4j:inference4j-genai'
    ```

=== "Maven"

    ```xml
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-genai</artifactId>
    </dependency>
    ```

This is a separate module backed by onnxruntime-genai. See the [Generative AI guide](../generative-ai/introduction.md) for details.

!!! warning "Don't override the ONNX Runtime version alongside `inference4j-genai`"

    `inference4j-genai` pulls in a fat JAR that bundles ONNX Runtime's native
    libraries under the same resource path the official ONNX Runtime JAR uses.
    `inference4j-core` pins the version those natives were built against, so
    forcing a different ONNX Runtime version can silently load the wrong native
    library. See [Generative AI &rarr; Versions](../generative-ai/introduction.md#versions).

## JVM flags

ONNX Runtime requires native access. Add this flag to your JVM arguments:

```
--enable-native-access=ALL-UNNAMED
```

Or, if you're on the module path:

```
--enable-native-access=com.microsoft.onnxruntime
```

### Setting JVM flags in Gradle

```groovy
tasks.withType(JavaExec).configureEach {
    jvmArgs '--enable-native-access=ALL-UNNAMED'
}

tasks.withType(Test).configureEach {
    jvmArgs '--enable-native-access=ALL-UNNAMED'
}
```

## Spring Boot

For Spring Boot applications, use the starter instead. It requires **Spring Boot 4.0 or
later** as of inference4j 0.11.0 — see the [Spring Boot guide](../guides/spring-boot.md#requirements)
if you are still on Boot 3.

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

See the [Spring Boot guide](../guides/spring-boot.md) for configuration details.

## GPU support

The default dependency includes CPU and CoreML (macOS) support. For CUDA (Linux/Windows), swap the ONNX Runtime dependency:

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

See the [Hardware Acceleration guide](../guides/hardware-acceleration.md) for usage details.
