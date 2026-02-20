package io.github.inference4j.generation;

import io.github.inference4j.sampling.SamplingConfig;

import java.util.Set;

public record GenerationRequest(String prompt,
                                SamplingConfig sampling,
                                int maxNewTokens,
                                Set<String> stopSequences) {
}
