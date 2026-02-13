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

class WordPieceTokenizerTest {

    private static WordPieceTokenizer tokenizer;

    @BeforeAll
    static void setUp() {
        Path vocabPath = Path.of(Objects.requireNonNull(
                WordPieceTokenizerTest.class.getResource("/test-vocab.txt")).getPath());
        tokenizer = WordPieceTokenizer.fromVocabFile(vocabPath);
    }

    @Test
    void encode_simpleKnownWord() {
        EncodedInput result = tokenizer.encode("hello");
        // [CLS]=2, hello=5, [SEP]=3
        assertArrayEquals(new long[]{2, 5, 3}, result.inputIds());
    }

    @Test
    void encode_multipleKnownWords() {
        EncodedInput result = tokenizer.encode("hello world");
        // [CLS]=2, hello=5, world=6, [SEP]=3
        assertArrayEquals(new long[]{2, 5, 6, 3}, result.inputIds());
    }

    @Test
    void encode_subwordSplitting() {
        EncodedInput result = tokenizer.encode("testing");
        // "testing" -> "test" + "##ing"
        // [CLS]=2, test=7, ##ing=8, [SEP]=3
        assertArrayEquals(new long[]{2, 7, 8, 3}, result.inputIds());
    }

    @Test
    void encode_unknownWord() {
        EncodedInput result = tokenizer.encode("xyz");
        // [CLS]=2, [UNK]=1, [SEP]=3
        assertArrayEquals(new long[]{2, 1, 3}, result.inputIds());
    }

    @Test
    void encode_punctuationSplit() {
        EncodedInput result = tokenizer.encode("hello!");
        // [CLS]=2, hello=5, !=18, [SEP]=3
        assertArrayEquals(new long[]{2, 5, 18, 3}, result.inputIds());
    }

    @Test
    void encode_attentionMaskAllOnes() {
        EncodedInput result = tokenizer.encode("hello world");
        long[] expectedMask = {1, 1, 1, 1};
        assertArrayEquals(expectedMask, result.attentionMask());
    }

    @Test
    void encode_tokenTypeIdsAllZeros() {
        EncodedInput result = tokenizer.encode("hello world");
        long[] expectedTypes = {0, 0, 0, 0};
        assertArrayEquals(expectedTypes, result.tokenTypeIds());
    }

    @Test
    void encode_truncatesToMaxLength() {
        // "hello world test good morning" = 5 tokens + [CLS] + [SEP] = 7
        EncodedInput result = tokenizer.encode("hello world test good morning", 5);
        // Should truncate to 5 tokens total: [CLS] + 3 words + [SEP]
        assertEquals(5, result.inputIds().length);
        assertEquals(2, result.inputIds()[0]); // [CLS]
        assertEquals(3, result.inputIds()[4]); // [SEP] at end
    }

    @Test
    void encode_lowercasesInput() {
        EncodedInput result = tokenizer.encode("HELLO");
        // [CLS]=2, hello=5, [SEP]=3
        assertArrayEquals(new long[]{2, 5, 3}, result.inputIds());
    }

    @Test
    void encode_handlesMultipleSubwords() {
        EncodedInput result = tokenizer.encode("unbelievable");
        // "unbelievable" -> "un" + "##believ" + "##able"
        // [CLS]=2, un=13, ##believ=14, ##able=15, [SEP]=3
        assertArrayEquals(new long[]{2, 13, 14, 15, 3}, result.inputIds());
    }

    @Test
    void encode_emptyString() {
        EncodedInput result = tokenizer.encode("");
        // [CLS]=2, [SEP]=3
        assertArrayEquals(new long[]{2, 3}, result.inputIds());
    }
}
