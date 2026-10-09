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

import io.github.inference4j.exception.InputTooLongException;
import io.github.inference4j.tokenizer.EncodedInput;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TruncationGuardTest {

    private static EncodedInput kept(int keptTokens, int originalLength) {
        long[] ids = new long[keptTokens];
        long[] mask = new long[keptTokens];
        java.util.Arrays.fill(mask, 1L);
        return new EncodedInput(ids, mask, new long[keptTokens], null, originalLength);
    }

    @Test
    void failPolicyThrowsWithCountsWhenTruncated() {
        TruncationGuard guard = new TruncationGuard("MyTask", TruncationPolicy.FAIL);

        assertThatThrownBy(() -> guard.check(kept(512, 1843), 512))
                .isInstanceOfSatisfying(InputTooLongException.class, e -> {
                    assertThat(e.tokenCount()).isEqualTo(1843);
                    assertThat(e.maxTokens()).isEqualTo(512);
                    assertThat(e.getMessage()).contains("MyTask", "1843", "512");
                });
    }

    @Test
    void failPolicyAcceptsInputWithinLimit() {
        TruncationGuard guard = new TruncationGuard("MyTask", TruncationPolicy.FAIL);

        assertThatCode(() -> guard.check(kept(10, 10), 512)).doesNotThrowAnyException();
    }

    @Test
    void truncatePolicyDoesNotThrowWhenTruncated() {
        TruncationGuard guard = new TruncationGuard("MyTask", TruncationPolicy.TRUNCATE);

        assertThatCode(() -> {
            guard.check(kept(512, 1843), 512);
            guard.check(kept(512, 900), 512);
        }).doesNotThrowAnyException();
    }

    @Test
    void nullPolicyDefaultsToTruncate() {
        TruncationGuard guard = new TruncationGuard("MyTask", null);

        assertThat(guard.policy()).isEqualTo(TruncationPolicy.TRUNCATE);
        assertThatCode(() -> guard.check(kept(512, 1843), 512)).doesNotThrowAnyException();
    }

    @Test
    void paddingIsNotMistakenForTruncation() {
        // 77 positions, only 5 real tokens: padded, not truncated
        long[] mask = new long[77];
        java.util.Arrays.fill(mask, 0, 5, 1L);
        EncodedInput padded = new EncodedInput(new long[77], mask, new long[77], null, 5);
        TruncationGuard guard = new TruncationGuard("MyTask", TruncationPolicy.FAIL);

        assertThat(padded.truncated()).isFalse();
        assertThatCode(() -> guard.check(padded, 77)).doesNotThrowAnyException();
    }
}
