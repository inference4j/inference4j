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
package io.github.inference4j.genai;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GenerationResultTest {

    @Test
    void recordFieldsAreAccessible() {
        var result = new GenerationResult("Hello world", 5, 120);
        assertEquals("Hello world", result.text());
        assertEquals(5, result.tokenCount());
        assertEquals(120, result.durationMillis());
    }

    @Test
    void recordEquality() {
        var a = new GenerationResult("Hi", 2, 50);
        var b = new GenerationResult("Hi", 2, 50);
        assertEquals(a, b);
    }
}
