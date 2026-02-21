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

import java.util.Arrays;

public class TopPProcessor implements LogitsProcessor {

    private final float p;

    public TopPProcessor(float p) {
        this.p = p;
    }

    @Override
    public float[] process(float[] logits) {
        if (p >= 1.0f) return logits;

        float[] probs = MathOps.softmax(logits);

        int n = logits.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));

        float cumulative = 0;
        boolean[] keep = new boolean[n];
        for (int i = 0; i < n; i++) {
            cumulative += probs[indices[i]];
            keep[indices[i]] = true;
            if (cumulative >= p) break;
        }

        float[] result = logits.clone();
        for (int i = 0; i < n; i++) {
            if (!keep[i]) {
                result[i] = Float.NEGATIVE_INFINITY;
            }
        }
        return result;
    }
}
