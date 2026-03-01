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

package io.github.inference4j.generation;

import io.github.inference4j.sampling.SamplingConfig;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.assertj.core.api.Assertions.*;

class GenerationRecordsTest {

    @Test
    void generationRequest_fieldAccess() {
        SamplingConfig sampling = new SamplingConfig(0.7f, 50, 0.9f, 1.1f, 42L);
        Set<String> stopSequences = Set.of("<|end|>", "\n");
        GenerationRequest request = new GenerationRequest("Hello world", sampling, 128, stopSequences);

        assertThat(request.prompt()).isEqualTo("Hello world");
        assertThat(request.sampling()).isEqualTo(sampling);
        assertThat(request.maxNewTokens()).isEqualTo(128);
        assertThat(request.stopSequences()).containsExactlyInAnyOrder("<|end|>", "\n");
    }

    @Test
    void generationRequest_equality() {
        SamplingConfig sampling = new SamplingConfig(0.7f, 50, 0.9f, 1.1f, 42L);
        Set<String> stops = Set.of("</s>");
        GenerationRequest a = new GenerationRequest("prompt", sampling, 64, stops);
        GenerationRequest b = new GenerationRequest("prompt", sampling, 64, stops);

        assertThat(a).isEqualTo(b);
        assertThat(a.hashCode()).isEqualTo(b.hashCode());
    }

    @Test
    void generationRequest_inequality() {
        SamplingConfig sampling = new SamplingConfig(0.7f, 50, 0.9f, 1.1f, 42L);
        GenerationRequest a = new GenerationRequest("prompt A", sampling, 64, Set.of());
        GenerationRequest b = new GenerationRequest("prompt B", sampling, 64, Set.of());

        assertThat(a).isNotEqualTo(b);
    }

    @Test
    void forwardResult_fieldAccess() {
        float[] logits = {0.1f, 0.5f, 0.3f, 0.1f};
        ForwardResult result = new ForwardResult(logits);

        assertThat(result.logits()).isSameAs(logits);
    }

    @Test
    void chatTemplate_functionalInterface() {
        ChatTemplate template = userMessage -> "<|user|>" + userMessage + "<|end|>";

        assertThat(template.format("Hello")).isEqualTo("<|user|>Hello<|end|>");
    }

    @Test
    void chatTemplate_identity() {
        ChatTemplate template = userMessage -> userMessage;

        assertThat(template.format("raw input")).isEqualTo("raw input");
    }
}
