# Contributing to inference4j

Thanks for your interest in contributing! inference4j is in its early stages (0.x), so things move fast — but contributions are welcome and appreciated.

## Before You Start

- **Bug reports and feature requests** — open a [GitHub issue](https://github.com/inference4j/inference4j/issues). Include steps to reproduce for bugs, or a use case description for features.
- **Large changes** — new model wrappers, new modules, API changes, or anything that touches multiple modules: please open an issue first so we can discuss the approach before you invest time.
- **Questions** — open a [discussion](https://github.com/inference4j/inference4j/discussions) or ask in an issue.

## Submitting a Pull Request

1. **Fork** the repository and create a branch from `main`.
2. Make your changes in focused, logical commits. Each commit should represent a coherent unit of work.
3. **Include tests.** Every pull request must include tests that cover the new or changed behavior. PRs without tests will be asked to add them before review.
4. Ensure `./gradlew build` passes locally before pushing.
5. **Open the PR** against `main` with a clear title and description:
   - What does it do?
   - Why is the change needed?
   - Link to the related issue if there is one (use `Closes #123`).
6. CI will run automatically. All checks must pass before merge.

Keep pull requests focused. One feature or fix per PR makes review faster and history cleaner. If you find an unrelated issue along the way, open a separate PR for it.

## Testing Requirements

Tests are not optional. Every contribution must include appropriate test coverage:

- **Unit tests** — test post-processing logic, builder validation, and error paths with synthetic data. Use JUnit 5 and Mockito. Mock `InferenceSession` rather than loading real ONNX models.
- **Model integration tests** — contributions that add or modify model wrappers must pass the model test suite (`./gradlew modelTest`). These tests run real inference against hosted models and verify output correctness.
- **Examples** — new model wrappers must include a runnable example in `inference4j-examples` (see [Contributing a New Model](#contributing-a-new-model) below).

Look at existing tests for patterns. For example, `ResNetClassifierTest` shows how to test a vision wrapper: mock the session, test `postProcess()` with known logits, verify builder validation, and confirm close delegation.

## Contributing a New Model

Adding a model wrapper is one of the most impactful contributions you can make. Here's the process:

### 1. Open an issue first

Before writing code, open an issue with:

- **Model name and architecture** (e.g., "MobileNetV3 image classification")
- **Where the model is hosted** — HuggingFace URL, original paper, or source repository
- **Model license** — we can only accept models with permissive licenses (Apache 2.0, MIT, BSD, CC-BY, etc.)
- **ONNX availability** — is there an existing ONNX export, or does it need conversion?

We host all supported models under the [`inference4j`](https://huggingface.co/inference4j) HuggingFace organization to guarantee download stability and availability. During the issue discussion, we'll coordinate hosting the model with proper attribution to the original authors and license.

### 2. Implement the wrapper

A complete model contribution includes:

| What | Where |
|------|-------|
| Task wrapper class | `inference4j-tasks/src/main/java/io/github/inference4j/{domain}/` |
| Unit tests | `inference4j-tasks/src/test/java/io/github/inference4j/{domain}/` |
| Model integration test | `inference4j-tasks/src/test/java/io/github/inference4j/{domain}/` |
| Runnable example | `inference4j-examples/src/main/java/io/github/inference4j/examples/` |
| Preprocessing (if new) | `inference4j-preprocessing/src/main/java/io/github/inference4j/{domain}/` |

Follow the existing architecture. All task wrappers extend `AbstractInferenceTask<I, O>` and implement a domain interface (`ImageClassifier`, `ObjectDetector`, `SpeechRecognizer`, etc.). See [Adding a Wrapper](https://inference4j.github.io/inference4j/contributing/adding-a-wrapper/) in the docs for the full walkthrough.

### 3. Example required

Every new model must include a runnable example in `inference4j-examples`. The example should demonstrate the model's primary use case in the simplest way possible. See existing examples like `ImageClassificationExample.java` or `TextClassificationExample.java` for the pattern.

## Getting Started

**Requirements:** Java 17+, Gradle 9.2.1 (use the included wrapper).

```bash
./gradlew build          # Build all modules and run tests
./gradlew test           # Run unit tests only
./gradlew modelTest      # Run model integration tests (downloads models)
```

## Code Conventions

- **Java 17** — records, sealed interfaces, and pattern matching are encouraged
- **Apache 2.0 license header** — every Java file must start with the license header before the `package` declaration:
  ```java
  /*
   * Copyright 2026 the original author or authors.
   *
   * Licensed under the Apache License, Version 2.0 (the "License");
   * you may not use this file except in compliance with the License.
   * You may obtain a copy of the License at
   *
   *      https://www.apache.org/licenses/LICENSE-2.0
   *
   * Unless required by applicable law or agreed to in writing, software
   * distributed under the License is distributed on an "AS IS" BASIS,
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   * See the License for the specific language governing permissions and
   * limitations under the License.
   */
  ```
- **Packages by domain** — `io.github.inference4j.vision`, `io.github.inference4j.nlp`, `io.github.inference4j.audio` — not by module name
- **No unnecessary dependencies** — prefer JDK built-ins when possible
- **Minimal changes** — keep PRs focused. Don't reformat code you didn't change or add unrelated improvements

## Project Structure

| Module | Purpose |
|--------|---------|
| `inference4j-core` | Low-level ONNX Runtime abstractions (`InferenceSession`, `Tensor`, `MathOps`) |
| `inference4j-preprocessing` | Tokenizers, image transforms, audio processing |
| `inference4j-tasks` | Task-oriented model wrappers with domain-specific APIs |
| `inference4j-runtime` | Operational layer — routing, metrics |
| `inference4j-spring-boot-starter` | Spring Boot auto-configuration |
| `inference4j-examples` | Runnable examples |

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
