# Installation

## Requirements

- **Java 17** or higher
- ONNX Runtime 1.28.0 (included transitively via `inference4j-core`)

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

`inference4j-genai` includes `inference4j-core`, so the task wrappers are available too.

!!! note "ONNX Runtime version"

    onnxruntime-genai loads ONNX Runtime from the regular ONNX Runtime dependency that
    `inference4j-core` brings in. If you override the ONNX Runtime version, keep it at or
    above the one listed in [Generative AI &rarr; Versions](../generative-ai/introduction.md#versions).

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

## Pre-release snapshots (JitPack)

To try features merged to `main` before they reach Maven Central, build them through
[JitPack](https://jitpack.io/#inference4j/inference4j). JitPack serves every module under the
`com.github.inference4j.inference4j` group, at version `main-SNAPSHOT` (the latest commit on
`main`) or at a commit hash (a fixed build, recommended for anything you want to reproduce):

=== "Gradle"

    ```groovy
    repositories {
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }

    dependencies {
        implementation 'com.github.inference4j.inference4j:inference4j-core:main-SNAPSHOT'
    }
    ```

=== "Maven"

    ```xml
    <repositories>
        <repository>
            <id>jitpack.io</id>
            <url>https://jitpack.io</url>
        </repository>
    </repositories>

    <dependency>
        <groupId>com.github.inference4j.inference4j</groupId>
        <artifactId>inference4j-core</artifactId>
        <version>main-SNAPSHOT</version>
    </dependency>
    ```

The other modules work the same way (`inference4j-genai`, `inference4j-runtime`,
`inference4j-spring-boot-starter`); their dependencies on each other resolve from JitPack too.

!!! note "Snapshots"

    - Don't mix JitPack and Maven Central coordinates for inference4j modules in one build: the
      two groups would put the same classes on the classpath twice.
    - The first request for a new commit starts a build on JitPack, which can take a few minutes.
    - Gradle caches `-SNAPSHOT` versions for 24 hours; run with `--refresh-dependencies` to pick
      up a newer commit sooner.
    - Snapshots are not releases: APIs can change before the next version ships.

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
