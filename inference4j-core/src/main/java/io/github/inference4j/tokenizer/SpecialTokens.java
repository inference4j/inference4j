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

/**
 * The special token ids a {@code tokenizer.json} post-processor places around one text ({@code [CLS] A [SEP]}
 * for BERT-style models) or a pair of texts ({@code [CLS] A [SEP] B [SEP]}), for tokenizers that return the
 * texts' tokens without them.
 *
 * @param singlePrefix before a single text
 * @param singleSuffix after a single text
 * @param pairPrefix   before the first text of a pair
 * @param pairMiddle   between the two texts of a pair
 * @param pairSuffix   after the second text of a pair
 */
public record SpecialTokens(long[] singlePrefix, long[] singleSuffix,
                            long[] pairPrefix, long[] pairMiddle, long[] pairSuffix) {

    /** How many special tokens a single text gets. */
    public int singleCount() {
        return singlePrefix.length + singleSuffix.length;
    }

    /** How many special tokens a pair of texts gets. */
    public int pairCount() {
        return pairPrefix.length + pairMiddle.length + pairSuffix.length;
    }

    /** {@code singlePrefix + text + singleSuffix}. */
    public long[] wrap(long[] text) {
        return concat(singlePrefix, text, singleSuffix);
    }

    /** {@code pairPrefix + first + pairMiddle + second + pairSuffix}. */
    public long[] wrap(long[] first, long[] second) {
        return concat(pairPrefix, first, pairMiddle, second, pairSuffix);
    }

    private static long[] concat(long[]... parts) {
        int length = 0;
        for (long[] part : parts) {
            length += part.length;
        }
        long[] out = new long[length];
        int at = 0;
        for (long[] part : parts) {
            System.arraycopy(part, 0, out, at, part.length);
            at += part.length;
        }
        return out;
    }
}
