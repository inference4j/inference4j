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

package io.github.inference4j.text;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MiniLMRerankerTest {

    @Test
    void toScore_positiveLogit_highScore() {
        float score = MiniLMReranker.toScore(5.0f);
        assertTrue(score > 0.99f);
    }

    @Test
    void toScore_negativeLogit_lowScore() {
        float score = MiniLMReranker.toScore(-5.0f);
        assertTrue(score < 0.01f);
    }

    @Test
    void toScore_zeroLogit_returnsHalf() {
        float score = MiniLMReranker.toScore(0.0f);
        assertEquals(0.5f, score, 1e-5f);
    }

    @Test
    void toScore_outputBetweenZeroAndOne() {
        float[] testLogits = {-10f, -5f, -1f, 0f, 1f, 5f, 10f};
        for (float logit : testLogits) {
            float score = MiniLMReranker.toScore(logit);
            assertTrue(score > 0f && score < 1f,
                    "Score " + score + " for logit " + logit + " should be in (0, 1)");
        }
    }

    @Test
    void toScore_monotonicWithLogit() {
        float prev = MiniLMReranker.toScore(-10f);
        for (float logit = -9f; logit <= 10f; logit += 1f) {
            float current = MiniLMReranker.toScore(logit);
            assertTrue(current > prev,
                    "Score should increase monotonically with logit");
            prev = current;
        }
    }

    @Test
    void toScore_symmetricAroundZero() {
        float scorePlus = MiniLMReranker.toScore(3.0f);
        float scoreMinus = MiniLMReranker.toScore(-3.0f);
        assertEquals(1.0f, scorePlus + scoreMinus, 1e-5f);
    }
}
