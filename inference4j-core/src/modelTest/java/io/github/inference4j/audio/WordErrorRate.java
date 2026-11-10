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

package io.github.inference4j.audio;

import java.util.Locale;

/**
 * Word error rate: word-level edit distance (substitutions, insertions and deletions) divided by
 * the number of reference words. Case and punctuation are ignored, so {@code "Ask,"} matches
 * {@code "ask"}.
 */
final class WordErrorRate {

    private WordErrorRate() {
    }

    static double of(String reference, String hypothesis) {
        String[] ref = words(reference);
        String[] hyp = words(hypothesis);
        if (ref.length == 0) {
            return hyp.length == 0 ? 0.0 : 1.0;
        }
        int[] previous = new int[hyp.length + 1];
        int[] current = new int[hyp.length + 1];
        for (int j = 0; j <= hyp.length; j++) {
            previous[j] = j;
        }
        for (int i = 1; i <= ref.length; i++) {
            current[0] = i;
            for (int j = 1; j <= hyp.length; j++) {
                int substitution = previous[j - 1] + (ref[i - 1].equals(hyp[j - 1]) ? 0 : 1);
                current[j] = Math.min(substitution, Math.min(previous[j], current[j - 1]) + 1);
            }
            int[] swap = previous;
            previous = current;
            current = swap;
        }
        return (double) previous[hyp.length] / ref.length;
    }

    private static String[] words(String text) {
        String normalized = text.toLowerCase(Locale.ROOT).replaceAll("[^\\p{L}\\p{N}'\\s]", " ").trim();
        return normalized.isEmpty() ? new String[0] : normalized.split("\\s+");
    }
}
