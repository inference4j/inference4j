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

import java.util.Arrays;

public class TopKProcessor implements LogitsProcessor {

    private final int k;

    public TopKProcessor(int k) {
        this.k = k;
    }

    @Override
    public float[] process(float[] logits) {
        if (k <= 0 || k >= logits.length) return logits;

        float[] sorted = logits.clone();
        Arrays.sort(sorted);
        float threshold = sorted[sorted.length - k];

        float[] result = logits.clone();
        for (int i = 0; i < result.length; i++) {
            if (result[i] < threshold) {
                result[i] = Float.NEGATIVE_INFINITY;
            }
        }
        return result;
    }
}
