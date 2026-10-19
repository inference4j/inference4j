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

import io.github.inference4j.exception.InputTooLongException;
import io.github.inference4j.tokenizer.EncodedInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Applies a {@link TruncationPolicy} to tokenized input. Each task instance holds one guard,
 * so the truncation warning is logged once per instance; repeats are logged at DEBUG.
 */
public final class TruncationGuard {

    private static final Logger logger = LoggerFactory.getLogger(TruncationGuard.class);

    private final String task;
    private final TruncationPolicy policy;
    private final AtomicBoolean warned = new AtomicBoolean();

    /**
     * @param task   name used in messages, typically the task's simple class name
     * @param policy the policy to apply; {@code null} means {@link TruncationPolicy#TRUNCATE}
     */
    public TruncationGuard(String task, TruncationPolicy policy) {
        this.task = task;
        this.policy = policy != null ? policy : TruncationPolicy.TRUNCATE;
    }

    /**
     * Checks one encoded input against the policy.
     *
     * @param encoded   the tokenizer output
     * @param maxTokens the limit {@code encoded} was tokenized with
     * @throws InputTooLongException if the input was truncated and the policy is {@code FAIL}
     */
    public void check(EncodedInput encoded, int maxTokens) {
        check(encoded, maxTokens, "the remaining text was ignored");
    }

    /**
     * Like {@link #check(EncodedInput, int)}, with a description of which part of the input was kept,
     * for tasks that do not simply keep the first tokens.
     *
     * @param kept what the warning says about the truncated input, e.g. "kept the end of the message"
     */
    public void check(EncodedInput encoded, int maxTokens, String kept) {
        if (!encoded.truncated()) {
            return;
        }
        if (policy == TruncationPolicy.FAIL) {
            throw new InputTooLongException(task, encoded.originalLength(), maxTokens);
        }
        String message = "{}: input of {} tokens truncated to {}; " + kept + ". "
                + "Use truncation(TruncationPolicy.FAIL) to reject long input instead.";
        if (warned.compareAndSet(false, true)) {
            logger.warn(message, task, encoded.originalLength(), maxTokens);
        } else {
            logger.debug(message, task, encoded.originalLength(), maxTokens);
        }
    }

    /**
     * Checks an input measured in units other than tokens, such as words, against the policy.
     *
     * @param length the input's length
     * @param limit  the most the task accepts; longer input is cut to it
     * @param unit   the unit, plural (e.g. {@code "words"})
     * @return whether the input is over the limit (and, with {@code TRUNCATE}, will be cut)
     * @throws InputTooLongException if the input is over the limit and the policy is {@code FAIL}
     */
    public boolean check(int length, int limit, String unit) {
        if (length <= limit) {
            return false;
        }
        if (policy == TruncationPolicy.FAIL) {
            throw new InputTooLongException(task, length, limit, unit);
        }
        String message = "{}: input of {} " + unit + " truncated to {}; the remaining text was ignored. "
                + "Use truncation(TruncationPolicy.FAIL) to reject long input instead.";
        if (warned.compareAndSet(false, true)) {
            logger.warn(message, task, length, limit);
        } else {
            logger.debug(message, task, length, limit);
        }
        return true;
    }

    public TruncationPolicy policy() {
        return policy;
    }
}
