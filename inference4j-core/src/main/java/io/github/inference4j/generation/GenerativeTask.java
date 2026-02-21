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

import java.util.function.Consumer;

/**
 * A task that generates output autoregressively, producing tokens one at a time.
 *
 * <p>This is the generative counterpart to
 * {@link io.github.inference4j.InferenceTask InferenceTask}. While InferenceTask
 * performs a single forward pass, GenerativeTask runs an iterative generate loop.
 *
 * @param <I> the domain input type (e.g., {@code String}, {@code Path})
 * @param <O> the domain output type (e.g., {@code GenerationResult}, {@code Transcription})
 */
public interface GenerativeTask<I, O> extends AutoCloseable {

    /**
     * Generates output from the given input.
     *
     * @param input the domain input
     * @return the generation result
     */
    O generate(I input);

    /**
     * Generates output from the given input, streaming tokens to the listener
     * as they are produced.
     *
     * @param input         the domain input
     * @param tokenListener receives each decoded text fragment as it is generated
     * @return the complete generation result
     */
    O generate(I input, Consumer<String> tokenListener);
}
