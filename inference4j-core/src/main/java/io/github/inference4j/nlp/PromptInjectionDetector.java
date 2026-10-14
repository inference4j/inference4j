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
 * Detects prompt injection and jailbreak attempts in text, such as user input or retrieved documents,
 * before it reaches a language model.
 *
 * <pre>{@code
 * try (PromptInjectionDetector guard = DebertaPromptInjectionDetector.builder().build()) {
 *     InjectionCheck check = guard.check(userInput);
 *     if (check.isInjection()) {
 *         // reject the input
 *     }
 * }
 * }</pre>
 *
 * @see DebertaPromptInjectionDetector
 * @see InjectionCheck
 */
public interface PromptInjectionDetector extends AutoCloseable {

    /**
     * Checks the whole text: input longer than the model's limit is checked in overlapping windows,
     * so one injected passage flags the text.
     *
     * @param text the text to check
     * @return the injection probability and the verdict at the detector's threshold
     */
    InjectionCheck check(String text);

    @Override
    void close();
}
