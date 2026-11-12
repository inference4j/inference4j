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

package io.github.inference4j.tokenizer;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;

class Utf8AccumulatorTest {

    private static final byte[] ROCKET = "🚀".getBytes(StandardCharsets.UTF_8);   // 4 bytes

    @Test
    void holdsAFourByteCharacterSplitAcrossAdds() {
        Utf8Accumulator bytes = new Utf8Accumulator();
        bytes.add("a".getBytes(StandardCharsets.UTF_8), 1);
        bytes.add(ROCKET, 2);
        assertThat(bytes.takeComplete()).isEqualTo("a");

        bytes.add(new byte[]{ROCKET[2], ROCKET[3]}, 2);
        assertThat(bytes.takeComplete()).isEqualTo("🚀");
        assertThat(bytes.isEmpty()).isTrue();
    }

    @Test
    void takeAllReplacesAnUnfinishedCharacter() {
        Utf8Accumulator bytes = new Utf8Accumulator();
        bytes.add(ROCKET, 3);
        assertThat(bytes.takeAll()).isEqualTo("�");
        assertThat(bytes.isEmpty()).isTrue();
    }

    @Test
    void completeLengthFindsTheLastFullCharacter() {
        assertThat(Utf8Accumulator.completeLength(new byte[0])).isZero();
        assertThat(Utf8Accumulator.completeLength("ab".getBytes(StandardCharsets.UTF_8))).isEqualTo(2);
        assertThat(Utf8Accumulator.completeLength(new byte[]{'a', ROCKET[0], ROCKET[1]})).isEqualTo(1);
        assertThat(Utf8Accumulator.completeLength(ROCKET)).isEqualTo(4);
    }
}
