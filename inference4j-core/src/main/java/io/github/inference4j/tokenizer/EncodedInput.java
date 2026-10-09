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

package io.github.inference4j.tokenizer;

public record EncodedInput(
        long[] inputIds,
        long[] attentionMask,
        long[] tokenTypeIds,
        int[] wordIds,
        int originalLength
) {

    /**
     * Constructor for inputs that were not truncated: the original length is the number of
     * attended tokens.
     */
    public EncodedInput(long[] inputIds, long[] attentionMask, long[] tokenTypeIds, int[] wordIds) {
        this(inputIds, attentionMask, tokenTypeIds, wordIds, attendedTokens(attentionMask));
    }

    /**
     * Backward-compatible constructor without word IDs.
     */
    public EncodedInput(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) {
        this(inputIds, attentionMask, tokenTypeIds, null);
    }

    /**
     * Whether the tokenizer cut the input to fit {@code maxLength}: {@link #originalLength()}
     * (tokens before truncation, special tokens included) exceeds the tokens actually kept.
     * Padding is not counted as kept.
     */
    public boolean truncated() {
        return originalLength > attendedTokens(attentionMask);
    }

    private static int attendedTokens(long[] attentionMask) {
        int count = 0;
        for (long mask : attentionMask) {
            if (mask != 0) {
                count++;
            }
        }
        return count;
    }
}
