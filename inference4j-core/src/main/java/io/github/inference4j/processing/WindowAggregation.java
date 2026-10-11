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

import java.util.List;

/**
 * Combines the per-label scores of each window of a long input into one score per label.
 *
 * <p>Tasks that produce one result for a whole text, such as classification, process input longer
 * than the model's limit in overlapping windows (see {@link TokenWindows}). Each window yields a
 * score for every label, and a {@code WindowAggregation} decides what the text as a whole scores:
 *
 * <ul>
 *   <li>{@link #mean()}: the average across windows, an overall verdict for the text. One strongly
 *       scored passage is diluted by the rest.</li>
 *   <li>{@link #max()}: the highest score any window reached, so a label that fires anywhere in the
 *       text fires for the text. Suits moderation and multi-label tagging.</li>
 * </ul>
 *
 * <p>Any other rule can be supplied as a lambda, for example the median or a trimmed mean.
 */
@FunctionalInterface
public interface WindowAggregation {

    /**
     * Combines window scores into one score per label.
     *
     * @param windowScores one array per window, in text order, each holding a score per label
     *                     (probabilities after the model's softmax or sigmoid); never empty, and all
     *                     arrays have the same length
     * @return one score per label
     */
    float[] aggregate(List<float[]> windowScores);

    /**
     * The mean score of each label across all windows. With softmax scores the result is still a
     * probability distribution.
     */
    static WindowAggregation mean() {
        return windowScores -> {
            float[] result = new float[windowScores.get(0).length];
            for (float[] scores : windowScores) {
                for (int i = 0; i < result.length; i++) {
                    result[i] += scores[i];
                }
            }
            for (int i = 0; i < result.length; i++) {
                result[i] /= windowScores.size();
            }
            return result;
        };
    }

    /**
     * The highest score of each label in any window. With softmax scores the result no longer sums
     * to 1: each value is the strongest evidence for that label anywhere in the text.
     */
    static WindowAggregation max() {
        return windowScores -> {
            float[] result = windowScores.get(0).clone();
            for (float[] scores : windowScores) {
                for (int i = 0; i < result.length; i++) {
                    result[i] = Math.max(result[i], scores[i]);
                }
            }
            return result;
        };
    }
}
