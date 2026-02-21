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

package io.github.inference4j.nlp;

import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.generation.GenerativeTask;

/**
 * Generates text autoregressively from a text prompt.
 *
 * <p>This is the domain-level interface for text generation models such as
 * GPT-2, Phi, and Llama. Concrete implementations wire together tokenization,
 * a generative session, and sampling into a single callable.
 *
 * @see GenerativeTask
 * @see GenerationResult
 */
public interface TextGenerator extends GenerativeTask<String, GenerationResult> {
}
