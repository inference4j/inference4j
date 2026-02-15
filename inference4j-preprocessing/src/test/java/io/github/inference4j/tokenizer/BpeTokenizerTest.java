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

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;

class BpeTokenizerTest {

    private static BpeTokenizer tokenizer;

    @BeforeAll
    static void setUp() {
        Path vocabPath = Path.of(Objects.requireNonNull(
                BpeTokenizerTest.class.getResource("/test-bpe-vocab.json")).getPath());
        Path mergesPath = Path.of(Objects.requireNonNull(
                BpeTokenizerTest.class.getResource("/test-bpe-merges.txt")).getPath());
        tokenizer = BpeTokenizer.fromFiles(vocabPath, mergesPath);
    }

    @Test
    void encode_simpleWord() {
        // "hello" -> ['hell', 'o</w>'] -> IDs [10, 3]
        // Full: [BOS=100, 10, 3, EOS=101, padding...]
        EncodedInput result = tokenizer.encode("hello", 10);
        assertEquals(100, result.inputIds()[0], "BOS token");
        assertEquals(10, result.inputIds()[1], "hell");
        assertEquals(3, result.inputIds()[2], "o</w>");
        assertEquals(101, result.inputIds()[3], "EOS token");
    }

    @Test
    void encode_multipleWords() {
        // "hello world" -> ['hell', 'o</w>', 'w', 'orld</w>'] -> IDs [10, 3, 4, 13]
        // Full: [100, 10, 3, 4, 13, 101, 0, 0, 0, 0]
        EncodedInput result = tokenizer.encode("hello world", 10);
        assertArrayEquals(
                new long[]{100, 10, 3, 4, 13, 101, 0, 0, 0, 0},
                result.inputIds());
    }

    @Test
    void encode_singleCharacterWord() {
        // "a" -> ['a</w>'] -> IDs [14]
        // Full: [100, 14, 101, 0, ...]
        EncodedInput result = tokenizer.encode("a", 10);
        assertEquals(100, result.inputIds()[0]);
        assertEquals(14, result.inputIds()[1]);
        assertEquals(101, result.inputIds()[2]);
    }

    @Test
    void encode_emptyString() {
        // No tokens -> [BOS, EOS, padding...]
        EncodedInput result = tokenizer.encode("", 10);
        assertEquals(100, result.inputIds()[0]);
        assertEquals(101, result.inputIds()[1]);
        assertEquals(0, result.inputIds()[2]);
    }

    @Test
    void encode_attentionMask() {
        // "hello" = 4 tokens (BOS + hell + o</w> + EOS)
        EncodedInput result = tokenizer.encode("hello", 10);
        assertArrayEquals(
                new long[]{1, 1, 1, 1, 0, 0, 0, 0, 0, 0},
                result.attentionMask());
    }

    @Test
    void encode_tokenTypeIdsAllZeros() {
        EncodedInput result = tokenizer.encode("hello", 10);
        assertArrayEquals(new long[10], result.tokenTypeIds());
    }

    @Test
    void encode_padsToMaxLength() {
        EncodedInput result = tokenizer.encode("a", 10);
        assertEquals(10, result.inputIds().length);
        assertEquals(10, result.attentionMask().length);
        assertEquals(10, result.tokenTypeIds().length);
    }

    @Test
    void encode_truncatesToMaxLength() {
        // "hello world" = 6 tokens [BOS, hell, o</w>, w, orld</w>, EOS]
        // With maxLength=5, should truncate to [BOS, hell, o</w>, w, EOS]
        EncodedInput result = tokenizer.encode("hello world", 5);
        assertEquals(5, result.inputIds().length);
        assertEquals(100, result.inputIds()[0], "BOS preserved");
        assertEquals(101, result.inputIds()[4], "EOS at end");
    }

    @Test
    void encode_lowercasesInput() {
        // "HELLO" should produce same result as "hello"
        EncodedInput upper = tokenizer.encode("HELLO", 10);
        EncodedInput lower = tokenizer.encode("hello", 10);
        assertArrayEquals(upper.inputIds(), lower.inputIds());
    }

    @Test
    void encode_normalizesWhitespace() {
        // "hello  world" (double space) should produce same result as "hello world"
        EncodedInput doubleSpace = tokenizer.encode("hello  world", 10);
        EncodedInput singleSpace = tokenizer.encode("hello world", 10);
        assertArrayEquals(doubleSpace.inputIds(), singleSpace.inputIds());
    }

    @Test
    void encode_defaultMaxLength() {
        // Default max length is 77 (CLIP standard)
        EncodedInput result = tokenizer.encode("hello");
        assertEquals(77, result.inputIds().length);
    }
}
