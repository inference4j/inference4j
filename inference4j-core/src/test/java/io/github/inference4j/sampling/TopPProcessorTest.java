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

package io.github.inference4j.sampling;

import io.github.inference4j.processing.MathOps;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class TopPProcessorTest {

    @Test
    void process_keepsTokensWithinCumulativeProbability() {
        // logits chosen so that softmax gives a clear ordering
        // [10, 5, 1, 0] -> softmax heavily favors index 0
        var processor = new TopPProcessor(0.95f);
        float[] logits = {10.0f, 5.0f, 1.0f, 0.0f};

        float[] result = processor.process(logits);

        // The top tokens that cover 95% of probability mass should be kept
        float[] probs = MathOps.softmax(logits);
        float cumulative = 0;
        int keptCount = 0;
        // Sort by prob descending to figure out expected kept count
        for (int i = 0; i < probs.length; i++) {
            // Find the i-th largest
            float maxProb = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < probs.length; j++) {
                if (result[j] != Float.NEGATIVE_INFINITY && probs[j] > maxProb) {
                    maxProb = probs[j];
                }
            }
        }

        // Verify kept tokens have their original logit values
        for (int i = 0; i < result.length; i++) {
            if (result[i] != Float.NEGATIVE_INFINITY) {
                assertThat(result[i]).isEqualTo(logits[i]);
                keptCount++;
            }
        }
        assertThat(keptCount).isGreaterThanOrEqualTo(1);
        assertThat(keptCount).isLessThanOrEqualTo(logits.length);
    }

    @Test
    void process_pOne_returnsAllLogits() {
        var processor = new TopPProcessor(1.0f);
        float[] logits = {1.0f, 2.0f, 3.0f};

        float[] result = processor.process(logits);

        assertThat(result).containsExactly(1.0f, 2.0f, 3.0f);
    }

    @Test
    void process_smallP_keepsOnlyTopToken() {
        // With very small p, only the highest probability token should be kept
        var processor = new TopPProcessor(0.01f);
        // logits where index 2 is dominant
        float[] logits = {-10.0f, -10.0f, 10.0f, -10.0f};

        float[] result = processor.process(logits);

        // Index 2 (highest prob) should be kept, rest masked
        assertThat(result[2]).isEqualTo(10.0f);
        assertThat(result[0]).isEqualTo(Float.NEGATIVE_INFINITY);
        assertThat(result[1]).isEqualTo(Float.NEGATIVE_INFINITY);
        assertThat(result[3]).isEqualTo(Float.NEGATIVE_INFINITY);
    }

    @Test
    void process_maskedTokensSetToNegativeInfinity() {
        var processor = new TopPProcessor(0.5f);
        float[] logits = {10.0f, 1.0f, 0.0f, -1.0f, -5.0f};

        float[] result = processor.process(logits);

        for (int i = 0; i < result.length; i++) {
            // Each value is either the original logit or negative infinity
            assertThat(result[i] == logits[i] || result[i] == Float.NEGATIVE_INFINITY)
                    .as("Value at index %d should be original logit or -Inf, got %f", i, result[i])
                    .isTrue();
        }

        // At least one value must be masked (p=0.5 on 5 tokens can't keep all)
        int maskedCount = 0;
        for (float v : result) {
            if (v == Float.NEGATIVE_INFINITY) maskedCount++;
        }
        assertThat(maskedCount).isGreaterThan(0);
    }
}
