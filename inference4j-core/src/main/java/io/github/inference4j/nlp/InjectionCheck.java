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
 * The result of a {@link PromptInjectionDetector} check.
 *
 * @param probability the model's probability that the text contains a prompt injection, in [0, 1]
 * @param threshold   the probability at or above which the text counts as an injection
 */
public record InjectionCheck(float probability, float threshold) {

    /** Whether the text counts as a prompt injection: {@code probability >= threshold}. */
    public boolean isInjection() {
        return probability >= threshold;
    }
}
