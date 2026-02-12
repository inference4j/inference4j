# inference4j

inference4j is a modern, type-safe, and ergonomic AI inference library for Java, built on top of the ONNX Runtime. It aims to make integrating AI models into Java applications as simple as any other library, without sacrificing performance or type safety.

## Status: Design Phase
We are currently in the process of designing the architecture and API. No code is available yet.

## Documentation
- [Vision](docs/vision.md) - Why we are building inference4j and our core principles.
- [Architecture](docs/architecture.md) - How the project is structured.
- [API Design](docs/api-design.md) - Examples of the intended developer experience.
- [Roadmap](docs/roadmap.md) - Our plan for development.
- [Initial Brainstorm](docs/brainstorm-initial.md) - The original discussion that started the project.

## Project Structure (Planned)
- `inference4j-core`: Core abstractions and ONNX wrappers.
- `inference4j-models`: Handcrafted popular model implementations.
- `inference4j-preprocessing`: Data preparation utilities.
- `inference4j-codegen`: Type-safe code generator.
- `inference4j-spring-boot-starter`: Spring Boot integration.
