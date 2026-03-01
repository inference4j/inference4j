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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class TopKProcessorTest {

    @Test
    void process_keepsTopKValues() {
        var processor = new TopKProcessor(2);
        float[] logits = {1.0f, 5.0f, 3.0f, 2.0f};

        float[] result = processor.process(logits);

        // Top-2 values are 5.0 and 3.0 at indices 1 and 2
        assertThat(result[1]).isEqualTo(5.0f);
        assertThat(result[2]).isEqualTo(3.0f);
    }

    @Test
    void process_masksRemainingToNegativeInfinity() {
        var processor = new TopKProcessor(2);
        float[] logits = {1.0f, 5.0f, 3.0f, 2.0f};

        float[] result = processor.process(logits);

        // Indices 0 and 3 should be masked (values 1.0 and 2.0 are below threshold of 3.0)
        assertThat(result[0]).isEqualTo(Float.NEGATIVE_INFINITY);
        assertThat(result[3]).isEqualTo(Float.NEGATIVE_INFINITY);
    }

    @Test
    void process_kGreaterThanOrEqualLength_returnsAllLogits() {
        var processor = new TopKProcessor(5);
        float[] logits = {1.0f, 2.0f, 3.0f};

        float[] result = processor.process(logits);

        assertThat(result).containsExactly(1.0f, 2.0f, 3.0f);
    }

    @Test
    void process_kZero_returnsAllLogits() {
        var processor = new TopKProcessor(0);
        float[] logits = {1.0f, 2.0f, 3.0f};

        float[] result = processor.process(logits);

        assertThat(result).containsExactly(1.0f, 2.0f, 3.0f);
    }

    @Test
    void process_preservesOriginalOrder() {
        var processor = new TopKProcessor(3);
        float[] logits = {4.0f, 1.0f, 7.0f, 5.0f, 2.0f};

        float[] result = processor.process(logits);

        // Top-3: 7.0 (idx 2), 5.0 (idx 3), 4.0 (idx 0) — kept in original positions
        assertThat(result[0]).isEqualTo(4.0f);
        assertThat(result[2]).isEqualTo(7.0f);
        assertThat(result[3]).isEqualTo(5.0f);
        // Masked
        assertThat(result[1]).isEqualTo(Float.NEGATIVE_INFINITY);
        assertThat(result[4]).isEqualTo(Float.NEGATIVE_INFINITY);
    }
}
