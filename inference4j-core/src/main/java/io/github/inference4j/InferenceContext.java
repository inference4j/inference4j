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
 * Cross-stage data carrier for the inference pipeline.
 *
 * <p>Bundles the original domain input, the preprocessed tensor inputs, and the
 * raw tensor outputs into a single object that the {@link Postprocessor} can use
 * for context-aware post-processing.
 *
 * <p>This enables postprocessors to access data from earlier stages — for example,
 * a YOLO postprocessor can read the original image dimensions from {@link #input()}
 * to rescale bounding boxes back to pixel coordinates.
 *
 * @param input        the original domain input (e.g., {@code BufferedImage}, {@code String})
 * @param preprocessed the tensor inputs sent to the session (e.g., includes {@code attention_mask})
 * @param outputs      the tensor outputs from the session
 * @param <I>          the input type
 * @see AbstractInferenceTask
 */
public record InferenceContext<I>(
        I input,
        Map<String, Tensor> preprocessed,
        Map<String, Tensor> outputs
) {}
