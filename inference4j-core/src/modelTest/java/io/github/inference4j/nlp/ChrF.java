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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Corpus-level chrF (character n-gram F-score), in the style of sacreBLEU's default chrF2: character
 * n-grams of orders 1 to 6 with whitespace removed, statistics summed over the corpus, precision and recall
 * averaged over the orders, and recall weighted twice as much as precision. Scores are on a 0–100 scale.
 * Not guaranteed to match sacreBLEU to the last digit.
 */
final class ChrF {

    private static final int MAX_ORDER = 6;
    private static final double BETA = 2;

    private ChrF() {
    }

    /** chrF of one hypothesis against its reference. */
    static double of(String reference, String hypothesis) {
        return corpus(List.of(reference), List.of(hypothesis));
    }

    /** chrF of a corpus: {@code hypotheses.get(i)} is scored against {@code references.get(i)}. */
    static double corpus(List<String> references, List<String> hypotheses) {
        if (references.size() != hypotheses.size()) {
            throw new IllegalArgumentException("references and hypotheses differ in size");
        }
        long[] matches = new long[MAX_ORDER + 1];
        long[] hypothesisCounts = new long[MAX_ORDER + 1];
        long[] referenceCounts = new long[MAX_ORDER + 1];
        for (int i = 0; i < references.size(); i++) {
            String reference = references.get(i).replaceAll("\\s+", "");
            String hypothesis = hypotheses.get(i).replaceAll("\\s+", "");
            for (int n = 1; n <= MAX_ORDER; n++) {
                Map<String, Integer> referenceGrams = grams(reference, n);
                Map<String, Integer> hypothesisGrams = grams(hypothesis, n);
                for (Map.Entry<String, Integer> gram : hypothesisGrams.entrySet()) {
                    matches[n] += Math.min(gram.getValue(), referenceGrams.getOrDefault(gram.getKey(), 0));
                }
                hypothesisCounts[n] += Math.max(0, hypothesis.length() - n + 1);
                referenceCounts[n] += Math.max(0, reference.length() - n + 1);
            }
        }
        double precision = 0;
        double recall = 0;
        int orders = 0;
        for (int n = 1; n <= MAX_ORDER; n++) {
            if (hypothesisCounts[n] == 0 || referenceCounts[n] == 0) {
                continue;
            }
            precision += (double) matches[n] / hypothesisCounts[n];
            recall += (double) matches[n] / referenceCounts[n];
            orders++;
        }
        if (orders == 0) {
            return 0;
        }
        precision /= orders;
        recall /= orders;
        if (precision == 0 && recall == 0) {
            return 0;
        }
        double beta2 = BETA * BETA;
        return 100 * (1 + beta2) * precision * recall / (beta2 * precision + recall);
    }

    private static Map<String, Integer> grams(String text, int n) {
        Map<String, Integer> grams = new HashMap<>();
        for (int i = 0; i + n <= text.length(); i++) {
            grams.merge(text.substring(i, i + n), 1, Integer::sum);
        }
        return grams;
    }
}
