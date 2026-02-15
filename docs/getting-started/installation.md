# Installation

## Requirements

- **Java 17** or higher
- ONNX Runtime (included transitively)

## Add the dependency

`inference4j-tasks` is the only dependency you need — it transitively includes core, preprocessing, and runtime.

=== "Gradle"

    ```groovy
    implementation 'io.github.inference4j:inference4j-tasks:${inference4jVersion}'
    ```

=== "Maven"

    ```xml
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-tasks</artifactId>
        <version>${inference4jVersion}</version>
    </dependency>
    ```

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

For Spring Boot applications, use the starter instead:

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
    implementation('io.github.inference4j:inference4j-tasks:${inference4jVersion}') {
        exclude group: 'com.microsoft.onnxruntime', module: 'onnxruntime'
    }
    implementation 'com.microsoft.onnxruntime:onnxruntime_gpu:${onnxruntimeVersion}'
    ```

=== "Maven"

    ```xml
    <dependency>
        <groupId>io.github.inference4j</groupId>
        <artifactId>inference4j-tasks</artifactId>
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
