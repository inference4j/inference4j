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

import java.util.concurrent.ThreadLocalRandom;

public class CategoricalSampler implements LogitsSampler {

    @Override
    public int sample(float[] logits) {
        var probs = MathOps.softmax(logits);
        var random = ThreadLocalRandom.current().nextFloat();
        var sum = 0f;
        for (int i = 0; i < probs.length; i++) {
            sum += probs[i];
            if (sum >= random) {
                return i;
            }
        }
        return probs.length - 1;
    }
}
