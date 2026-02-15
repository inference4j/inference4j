# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in inference4j, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/inference4j/inference4j/security)
2. Click **"Report a vulnerability"**
3. Fill in the details (description, steps to reproduce, potential impact)

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation plan within 7 days.

## Scope

inference4j runs AI models locally — there are no API keys, remote services, or network calls during inference. The primary security surface is:

- **Model loading** — models are downloaded from HuggingFace over HTTPS and cached locally
- **ONNX Runtime native code** — inference4j depends on ONNX Runtime's native library
- **Input processing** — image, audio, and text preprocessing

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x (latest) | Yes |
