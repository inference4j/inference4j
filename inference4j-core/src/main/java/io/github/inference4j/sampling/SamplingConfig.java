package io.github.inference4j.sampling;

public record SamplingConfig(float temperature,
                             int topK,
                             float topP,
                             float repetitionPenalty,
                             long seed) {
}
