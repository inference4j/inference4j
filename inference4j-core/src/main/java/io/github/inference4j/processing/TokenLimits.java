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

package io.github.inference4j.processing;

/**
 * The token budget of one task of a generative model: how much input one call accepts and how much
 * output it produces. Long-input strategies use it to cut text into pieces the model takes whole.
 *
 * <p>{@code text} is processed whole if and only if
 * {@code countInputTokens(text) <= maxInputTokens()}.
 */
public interface TokenLimits {

    /**
     * The most tokens one call accepts, task prompt and special tokens included;
     * {@link Integer#MAX_VALUE} when the limit is unknown.
     */
    int maxInputTokens();

    /** The most tokens one call generates. */
    int maxOutputTokens();

    /**
     * The tokens the model would receive for {@code text}: the task's prompt prefix, the text itself and
     * the special tokens added around it. Never truncates.
     */
    int countInputTokens(String text);
}
