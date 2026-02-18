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
package io.github.inference4j.vision;

import java.nio.file.Path;

/**
 * Input for a vision-language model: an image and a text prompt.
 *
 * <p>Used by {@link VisionLanguageModel} to pair an image with a question
 * or instruction. You typically don't need to create this directly — use
 * the convenience methods {@code describe()} and {@code ask()} instead.
 *
 * @param imagePath path to an image file (PNG, JPEG, etc.)
 * @param prompt    the text prompt or question about the image
 * @see VisionLanguageModel#describe(Path)
 * @see VisionLanguageModel#ask(Path, String)
 */
public record VisionInput(Path imagePath, String prompt) {
}
