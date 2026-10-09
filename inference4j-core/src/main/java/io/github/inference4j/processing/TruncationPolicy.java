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
 * What a task does with input longer than its model's token limit.
 *
 * <p>Tasks that support processing long input in full (for example, overlapping windows)
 * expose that as a separate builder option; this policy applies when no such strategy is
 * configured.
 */
public enum TruncationPolicy {

    /**
     * Keep the first tokens up to the limit and drop the rest. A warning is logged the first
     * time a task instance truncates input. This is the default.
     */
    TRUNCATE,

    /**
     * Reject the input with an {@link io.github.inference4j.exception.InputTooLongException}.
     * Useful in tests and pipelines where losing part of the input is unacceptable.
     */
    FAIL
}
