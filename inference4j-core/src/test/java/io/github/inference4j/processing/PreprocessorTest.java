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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class PreprocessorTest {

    @Test
    void identity_returnsInput() {
        Preprocessor<String, String> identity = Preprocessor.identity();
        assertThat(identity.process("hello")).isEqualTo("hello");
    }

    @Test
    void identity_withInteger() {
        Preprocessor<Integer, Integer> identity = Preprocessor.identity();
        assertThat(identity.process(42)).isEqualTo(42);
    }

    @Test
    void andThen_composesInOrder() {
        Preprocessor<String, Integer> length = String::length;
        Preprocessor<Integer, String> toString = Object::toString;

        Preprocessor<String, String> pipeline = length.andThen(toString);

        assertThat(pipeline.process("hello")).isEqualTo("5");
    }

    @Test
    void identity_andThen_isIdempotent() {
        Preprocessor<String, String> identity = Preprocessor.<String>identity().andThen(Preprocessor.identity());
        assertThat(identity.process("test")).isEqualTo("test");
    }

    @Test
    void andThen_threeStageChain() {
        Preprocessor<String, String> trim = String::trim;
        Preprocessor<String, String> upperCase = String::toUpperCase;
        Preprocessor<String, Integer> length = String::length;

        Preprocessor<String, Integer> pipeline = trim.andThen(upperCase).andThen(length);

        assertThat(pipeline.process("  hi  ")).isEqualTo(2);
    }
}
