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

import java.util.ArrayList;
import java.util.List;

/**
 * Splits a long token sequence into overlapping windows that fit a model's input limit, so
 * token-level tasks (one prediction per token) can process input of any length.
 *
 * <p>Each token is <em>owned</em> by exactly one window: the one in which it sits most centrally.
 * The boundary between two neighbouring windows is the midpoint of their overlap, so every owned
 * token has at least {@code stride / 2} tokens of context on each side (except at the start and end
 * of the whole sequence). A task runs the model on every window and keeps, for each token, the
 * prediction from the window that owns it.
 *
 * <pre>{@code
 * for (TokenWindows.Window w : TokenWindows.plan(tokens.length, 510, 128)) {
 *     // run the model on tokens[w.start() .. w.end())
 *     // keep predictions for positions [w.ownedStart(), w.ownedEnd())
 * }
 * }</pre>
 */
public final class TokenWindows {

    /**
     * A window over token positions {@code [start, end)} whose predictions are used for positions
     * {@code [ownedStart, ownedEnd)}.
     */
    public record Window(int start, int end, int ownedStart, int ownedEnd) {
    }

    private TokenWindows() {
    }

    /**
     * Plans windows over {@code tokenCount} tokens.
     *
     * <p>Windows hold {@code windowSize} tokens and start every {@code windowSize - stride} tokens.
     * The last window is aligned to the end of the sequence so it is always full, which can make its
     * overlap with the previous window larger than {@code stride}. Owned ranges are contiguous,
     * disjoint, and together cover {@code [0, tokenCount)}. A sequence that fits in one window
     * produces a single window owning everything.
     *
     * @param tokenCount number of tokens to cover
     * @param windowSize maximum tokens per window
     * @param stride     number of tokens shared by consecutive windows; {@code 0 <= stride < windowSize}
     * @throws IllegalArgumentException for a negative token count, non-positive window size, or a
     *                                  stride outside {@code [0, windowSize)}
     */
    public static List<Window> plan(int tokenCount, int windowSize, int stride) {
        if (tokenCount < 0) {
            throw new IllegalArgumentException("tokenCount must not be negative, got " + tokenCount);
        }
        if (windowSize <= 0) {
            throw new IllegalArgumentException("windowSize must be positive, got " + windowSize);
        }
        if (stride < 0 || stride >= windowSize) {
            throw new IllegalArgumentException(
                    "stride must be in [0, windowSize), got " + stride + " for windowSize " + windowSize);
        }
        if (tokenCount <= windowSize) {
            return List.of(new Window(0, tokenCount, 0, tokenCount));
        }

        List<Integer> starts = new ArrayList<>();
        int step = windowSize - stride;
        for (int start = 0; ; start += step) {
            if (start + windowSize >= tokenCount) {
                starts.add(tokenCount - windowSize);    // align the last window to the end
                break;
            }
            starts.add(start);
        }

        List<Window> windows = new ArrayList<>(starts.size());
        int ownedStart = 0;
        for (int i = 0; i < starts.size(); i++) {
            int start = starts.get(i);
            int end = start + windowSize;
            int ownedEnd = i == starts.size() - 1
                    ? tokenCount
                    : (starts.get(i + 1) + end) / 2;    // midpoint of the overlap with the next window
            windows.add(new Window(start, end, ownedStart, ownedEnd));
            ownedStart = ownedEnd;
        }
        return List.copyOf(windows);
    }
}
