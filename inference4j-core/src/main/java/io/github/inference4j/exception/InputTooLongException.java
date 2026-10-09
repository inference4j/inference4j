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
package io.github.inference4j.exception;

/**
 * Thrown when input exceeds a task's token limit and the task is configured with
 * {@link io.github.inference4j.processing.TruncationPolicy#FAIL}.
 */
public class InputTooLongException extends InferenceException {

    private final int tokenCount;
    private final int maxTokens;

    public InputTooLongException(String task, int tokenCount, int maxTokens) {
        super(task + ": input of " + tokenCount + " tokens exceeds the limit of " + maxTokens + " tokens");
        this.tokenCount = tokenCount;
        this.maxTokens = maxTokens;
    }

    /** Number of tokens the input produced, including special tokens. */
    public int tokenCount() {
        return tokenCount;
    }

    /** The task's token limit, including special tokens. */
    public int maxTokens() {
        return maxTokens;
    }
}
