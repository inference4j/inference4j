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

package io.github.inference4j;

import java.util.Map;

/**
 * Result of the preprocessing stage, carrying both the tensor inputs for the ONNX session
 * and optional metadata for the postprocessor.
 *
 * <p>Metadata allows the preprocessor to pass arbitrary data to the postprocessor without
 * it being sent to the ONNX session. For example, a NER preprocessor can pass word IDs
 * for subword-to-word alignment.
 *
 * <h2>Without metadata (most tasks)</h2>
 * <pre>{@code
 * // Preprocessor returns tensors only — metadata defaults to empty map
 * text -> {
 *     EncodedInput encoded = tokenizer.encode(text, maxLength);
 *     long[] shape = {1, encoded.inputIds().length};
 *     Map<String, Tensor> inputs = Map.of(
 *             "input_ids", Tensor.fromLongs(encoded.inputIds(), shape),
 *             "attention_mask", Tensor.fromLongs(encoded.attentionMask(), shape));
 *     return PreprocessResult.of(inputs);
 * }
 * }</pre>
 *
 * <h2>With metadata (e.g., NER word IDs)</h2>
 * <pre>{@code
 * // Preprocessor passes word IDs as metadata for the postprocessor
 * text -> {
 *     EncodedInput encoded = tokenizer.encode(text, maxLength);
 *     long[] shape = {1, encoded.inputIds().length};
 *     Map<String, Tensor> inputs = Map.of(
 *             "input_ids", Tensor.fromLongs(encoded.inputIds(), shape),
 *             "attention_mask", Tensor.fromLongs(encoded.attentionMask(), shape));
 *     return PreprocessResult.of(inputs, Map.of("wordIds", encoded.wordIds()));
 * }
 *
 * // Postprocessor reads metadata
 * ctx -> {
 *     int[] wordIds = (int[]) ctx.metadata().get("wordIds");
 *     // ... use wordIds for subword-to-word alignment
 * }
 * }</pre>
 *
 * @see AbstractInferenceTask
 * @see InferenceContext
 */
public record PreprocessResult(
        Map<String, Tensor> tensors,
        Map<String, Object> metadata
) {

    /**
     * Creates a result with tensor inputs and no metadata.
     *
     * @param tensors the tensor inputs for the ONNX session
     * @return a new preprocess result
     */
    public static PreprocessResult of(Map<String, Tensor> tensors) {
        return new PreprocessResult(tensors, Map.of());
    }

    /**
     * Creates a result with tensor inputs and metadata.
     *
     * @param tensors  the tensor inputs for the ONNX session
     * @param metadata arbitrary data to pass to the postprocessor
     * @return a new preprocess result
     */
    public static PreprocessResult of(Map<String, Tensor> tensors, Map<String, Object> metadata) {
        return new PreprocessResult(tensors, metadata);
    }
}
