package io.github.inference4j.generation;

import java.time.Duration;

public record GenerationResult(String text,
                               int promptTokens,
                               int generatedTokens,
                               Duration duration) {
}
