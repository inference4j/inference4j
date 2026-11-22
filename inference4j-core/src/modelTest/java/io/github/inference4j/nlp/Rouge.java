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
import java.util.Locale;
import java.util.Map;

/**
 * ROUGE-1, ROUGE-2 and ROUGE-L F1, as Google's {@code rouge_score} computes them without stemming: text is
 * lower-cased and split on every character outside {@code [a-z0-9]}. Scores are 0–100.
 */
final class Rouge {

    /** F1 scores, plus ROUGE-1's precision and recall, which show how output length shapes the F1. */
    record Scores(double rouge1, double rouge2, double rougeL, double rouge1Precision, double rouge1Recall) {
    }

    private Rouge() {
    }

    /** The mean of each score over the pairs, as {@code rouge_score}'s aggregation reports it. */
    static Scores mean(List<String> references, List<String> hypotheses) {
        double r1 = 0;
        double r2 = 0;
        double rl = 0;
        double p1 = 0;
        double c1 = 0;
        for (int i = 0; i < references.size(); i++) {
            Scores s = score(references.get(i), hypotheses.get(i));
            r1 += s.rouge1();
            r2 += s.rouge2();
            rl += s.rougeL();
            p1 += s.rouge1Precision();
            c1 += s.rouge1Recall();
        }
        int n = references.size();
        return new Scores(r1 / n, r2 / n, rl / n, p1 / n, c1 / n);
    }

    static Scores score(String reference, String hypothesis) {
        List<String> ref = tokens(reference);
        List<String> hyp = tokens(hypothesis);
        int unigramOverlap = overlap(ngrams(ref, 1), ngrams(hyp, 1));
        return new Scores(100 * rougeN(ref, hyp, 1), 100 * rougeN(ref, hyp, 2), 100 * rougeL(ref, hyp),
                hyp.isEmpty() ? 0 : 100.0 * unigramOverlap / hyp.size(),
                ref.isEmpty() ? 0 : 100.0 * unigramOverlap / ref.size());
    }

    static List<String> tokens(String text) {
        String cleaned = text.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", " ").strip();
        return cleaned.isEmpty() ? List.of() : List.of(cleaned.split(" "));
    }

    private static double rougeN(List<String> ref, List<String> hyp, int n) {
        int overlap = overlap(ngrams(ref, n), ngrams(hyp, n));
        return f1(overlap, Math.max(0, hyp.size() - n + 1), Math.max(0, ref.size() - n + 1));
    }

    private static int overlap(Map<String, Integer> refGrams, Map<String, Integer> hypGrams) {
        int overlap = 0;
        for (Map.Entry<String, Integer> gram : hypGrams.entrySet()) {
            overlap += Math.min(gram.getValue(), refGrams.getOrDefault(gram.getKey(), 0));
        }
        return overlap;
    }

    private static double rougeL(List<String> ref, List<String> hyp) {
        int[] previous = new int[hyp.size() + 1];
        for (String r : ref) {
            int[] current = new int[hyp.size() + 1];
            for (int j = 1; j <= hyp.size(); j++) {
                current[j] = r.equals(hyp.get(j - 1)) ? previous[j - 1] + 1 : Math.max(previous[j], current[j - 1]);
            }
            previous = current;
        }
        return f1(previous[hyp.size()], hyp.size(), ref.size());
    }

    private static Map<String, Integer> ngrams(List<String> tokens, int n) {
        Map<String, Integer> grams = new HashMap<>();
        for (int i = 0; i + n <= tokens.size(); i++) {
            grams.merge(String.join(" ", tokens.subList(i, i + n)), 1, Integer::sum);
        }
        return grams;
    }

    private static double f1(int overlap, int hypothesisCount, int referenceCount) {
        if (overlap == 0 || hypothesisCount == 0 || referenceCount == 0) {
            return 0;
        }
        double precision = (double) overlap / hypothesisCount;
        double recall = (double) overlap / referenceCount;
        return 2 * precision * recall / (precision + recall);
    }
}
