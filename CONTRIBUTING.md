# Contributing to inference4j

Thanks for your interest in contributing! This project is in its early stages, so things move fast — but contributions are welcome.

## Getting Started

**Requirements:** Java 17, Gradle 9.2.1 (use the included wrapper).

```bash
./gradlew build    # Build all modules and run tests
./gradlew test     # Run tests only
```

## How to Contribute

1. Fork the repository and create a branch from `main`
2. Make your changes
3. Ensure `./gradlew build` passes
4. Open a pull request with a clear description of what and why

For larger changes (new model wrappers, new modules, API changes), please open an issue first to discuss the approach.

## Code Conventions

- **Java 17** — records and pattern matching are encouraged
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
- **Packages by domain** — packages are organized by feature (`io.github.inference4j.vision`, `io.github.inference4j.nlp`, `io.github.inference4j.audio`), not by module name
- **Tests** — JUnit 5, test post-processing logic with synthetic data (no ONNX model files in tests)
- **No unnecessary dependencies** — prefer JDK built-ins when possible

## Project Structure

| Module | Purpose |
|--------|---------|
| `inference4j-core` | Low-level ONNX Runtime abstractions (`InferenceSession`, `Tensor`, `MathOps`) |
| `inference4j-preprocessing` | Tokenizers, image transforms, audio processing |
| `inference4j-tasks` | Task-oriented inference wrappers with domain-specific APIs |
| `inference4j-runtime` | Operational layer — routing, metrics |
| `inference4j-examples` | Runnable examples |

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
