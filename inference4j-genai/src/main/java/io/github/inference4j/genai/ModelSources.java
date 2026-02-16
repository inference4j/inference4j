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
package io.github.inference4j.genai;

import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;

import java.util.List;

/**
 * Factory for preconfigured {@link GenerativeModel} instances for generative AI models.
 *
 * <p>Each factory method returns a {@link GenerativeModel} that encapsulates the model
 * repository, required files, download logic, and chat template. Use these with
 * {@link io.github.inference4j.nlp.TextGenerator}:
 *
 * <pre>{@code
 * try (var gen = TextGenerator.builder()
 *         .model(ModelSources.phi3Mini())
 *         .build()) {
 *     gen.generate("Hello!");
 * }
 * }</pre>
 *
 * @see io.github.inference4j.nlp.TextGenerator
 */
public final class ModelSources {

    private ModelSources() {
    }

    /**
     * Phi-3 Mini 4K Instruct (INT4, ~2.7 GB).
     *
     * <p>3.8B-parameter lightweight model from Microsoft, quantized to INT4
     * for CPU inference. Hosted at {@code inference4j/phi-3-mini-4k-instruct}.
     *
     * @return a preconfigured generative model for Phi-3 Mini
     */
    public static GenerativeModel phi3Mini() {
        return new GenerativeModel(
                preconfigured("inference4j/phi-3-mini-4k-instruct", List.of(
                        "genai_config.json",
                        "config.json",
                        "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
                        "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
                        "tokenizer.json",
                        "tokenizer.model",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "added_tokens.json"
                )),
                message -> "<|user|>\n" + message + "<|end|>\n<|assistant|>\n"
        );
    }

    /**
     * DeepSeek-R1-Distill-Qwen-1.5B (INT4, ~1 GB).
     *
     * <p>1.5B-parameter reasoning model distilled from DeepSeek-R1, quantized
     * to INT4 for CPU inference. Hosted at {@code inference4j/deepseek-r1-distill-qwen-1.5b}.
     *
     * @return a preconfigured generative model for DeepSeek-R1 1.5B
     */
    public static GenerativeModel deepSeekR1_1_5B() {
        return new GenerativeModel(
                preconfigured("inference4j/deepseek-r1-distill-qwen-1.5b", List.of(
                        "genai_config.json",
                        "model.onnx",
                        "model.onnx.data",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json"
                )),
                message -> "<\uFF5Cbegin\u2581of\u2581sentence\uFF5C>"
                        + "<\uFF5CUser\uFF5C>" + message + "<\uFF5CAssistant\uFF5C>"
        );
    }

    private static ModelSource preconfigured(String repoId, List<String> requiredFiles) {
        return modelId -> HuggingFaceModelSource.defaultInstance()
                .resolve(repoId, requiredFiles);
    }

}
