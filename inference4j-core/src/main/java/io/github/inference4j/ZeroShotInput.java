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

import java.util.List;

/**
 * Compound input for zero-shot classification — combines the primary input
 * with candidate labels to classify against.
 *
 * @param input           the primary input (e.g., an image)
 * @param candidateLabels the text labels to classify against
 * @param <I>             the primary input type
 */
public record ZeroShotInput<I>(I input, List<String> candidateLabels) {
}
