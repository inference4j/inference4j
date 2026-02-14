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

/**
 * A single text classification result with label, class index, and confidence score.
 *
 * @param label      the predicted class label (e.g., "POSITIVE", "NEGATIVE")
 * @param index      the class index in the model's output
 * @param confidence the confidence score (probability after activation)
 */
public record TextClassification(String label, int index, float confidence) {
}
