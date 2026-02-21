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

package io.github.inference4j.generation;

import java.util.Set;
import java.util.function.Consumer;

/**
 * Buffers decoded token fragments to ensure stop sequences are never emitted
 * to the listener. Maintains a sliding window of at most {@code maxStopLength}
 * characters, flushing confirmed-safe text as soon as possible.
 *
 * <p>All string operations are bounded by the longest stop sequence length,
 * giving O(1) per-token cost regardless of total generated length.
 */
class TokenStreamer {

    private final Set<String> stopSequences;
    private final Consumer<String> listener;
    private final int maxStopLength;
    private final StringBuilder buffer = new StringBuilder();
    private final StringBuilder flushed = new StringBuilder();
    private boolean stopped;

    TokenStreamer(Set<String> stopSequences, Consumer<String> listener) {
        this.stopSequences = stopSequences;
        this.listener = listener;
        this.maxStopLength = stopSequences.stream()
                .mapToInt(String::length)
                .max()
                .orElse(0);
    }

    void accept(String fragment) {
        buffer.append(fragment);

        if (checkStop()) {
            return;
        }

        flushSafePrefix();
    }

    void flush() {
        if (stopped || buffer.isEmpty()) {
            return;
        }
        String remaining = buffer.toString();
        flushed.append(remaining);
        listener.accept(remaining);
        buffer.setLength(0);
    }

    boolean isStopped() {
        return stopped;
    }

    String getText() {
        return flushed.toString();
    }

    private boolean checkStop() {
        String bufferStr = buffer.toString();
        for (String stop : stopSequences) {
            int idx = bufferStr.indexOf(stop);
            if (idx >= 0) {
                // Flush everything before the stop sequence
                if (idx > 0) {
                    String before = bufferStr.substring(0, idx);
                    flushed.append(before);
                    listener.accept(before);
                }
                buffer.setLength(0);
                stopped = true;
                return true;
            }
        }
        return false;
    }

    private void flushSafePrefix() {
        int safeLength = buffer.length() - maxStopLength;
        if (safeLength > 0) {
            String safe = buffer.substring(0, safeLength);
            flushed.append(safe);
            listener.accept(safe);
            buffer.delete(0, safeLength);
        }
    }
}
