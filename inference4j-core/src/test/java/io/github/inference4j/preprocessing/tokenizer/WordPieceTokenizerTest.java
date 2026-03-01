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

package io.github.inference4j.preprocessing.tokenizer;

import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.WordPieceTokenizer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Objects;

import static org.assertj.core.api.Assertions.assertThat;

class WordPieceTokenizerTest {

    private static io.github.inference4j.tokenizer.WordPieceTokenizer tokenizer;

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
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 5, 3});
    }

    @Test
    void encode_multipleKnownWords() {
        EncodedInput result = tokenizer.encode("hello world");
        // [CLS]=2, hello=5, world=6, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 5, 6, 3});
    }

    @Test
    void encode_subwordSplitting() {
        EncodedInput result = tokenizer.encode("testing");
        // "testing" -> "test" + "##ing"
        // [CLS]=2, test=7, ##ing=8, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 7, 8, 3});
    }

    @Test
    void encode_unknownWord() {
        EncodedInput result = tokenizer.encode("xyz");
        // [CLS]=2, [UNK]=1, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 1, 3});
    }

    @Test
    void encode_punctuationSplit() {
        EncodedInput result = tokenizer.encode("hello!");
        // [CLS]=2, hello=5, !=18, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 5, 18, 3});
    }

    @Test
    void encode_attentionMaskAllOnes() {
        EncodedInput result = tokenizer.encode("hello world");
        assertThat(result.attentionMask()).isEqualTo(new long[]{1, 1, 1, 1});
    }

    @Test
    void encode_tokenTypeIdsAllZeros() {
        EncodedInput result = tokenizer.encode("hello world");
        assertThat(result.tokenTypeIds()).isEqualTo(new long[]{0, 0, 0, 0});
    }

    @Test
    void encode_truncatesToMaxLength() {
        // "hello world test good morning" = 5 tokens + [CLS] + [SEP] = 7
        EncodedInput result = tokenizer.encode("hello world test good morning", 5);
        // Should truncate to 5 tokens total: [CLS] + 3 words + [SEP]
        assertThat(result.inputIds().length).isEqualTo(5);
        assertThat(result.inputIds()[0]).isEqualTo(2); // [CLS]
        assertThat(result.inputIds()[4]).isEqualTo(3); // [SEP] at end
    }

    @Test
    void encode_lowercasesInput() {
        EncodedInput result = tokenizer.encode("HELLO");
        // [CLS]=2, hello=5, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 5, 3});
    }

    @Test
    void encode_handlesMultipleSubwords() {
        EncodedInput result = tokenizer.encode("unbelievable");
        // "unbelievable" -> "un" + "##believ" + "##able"
        // [CLS]=2, un=13, ##believ=14, ##able=15, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 13, 14, 15, 3});
    }

    @Test
    void encode_emptyString() {
        EncodedInput result = tokenizer.encode("");
        // [CLS]=2, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 3});
    }

    @Test
    void encodePair_basicSentencePair() {
        EncodedInput result = tokenizer.encode("hello", "world", 512);
        // [CLS]=2, hello=5, [SEP]=3, world=6, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 5, 3, 6, 3});
    }

    @Test
    void encodePair_tokenTypeIds() {
        EncodedInput result = tokenizer.encode("hello", "world", 512);
        // Segment A: [CLS](0) hello(0) [SEP](0), Segment B: world(1) [SEP](1)
        assertThat(result.tokenTypeIds()).isEqualTo(new long[]{0, 0, 0, 1, 1});
    }

    @Test
    void encodePair_attentionMaskAllOnes() {
        EncodedInput result = tokenizer.encode("hello", "world", 512);
        assertThat(result.attentionMask()).isEqualTo(new long[]{1, 1, 1, 1, 1});
    }

    @Test
    void encodePair_multipleTokens() {
        EncodedInput result = tokenizer.encode("hello world", "good morning", 512);
        // [CLS]=2, hello=5, world=6, [SEP]=3, good=11, morning=12, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 5, 6, 3, 11, 12, 3});
        assertThat(result.tokenTypeIds()).isEqualTo(new long[]{0, 0, 0, 0, 1, 1, 1});
    }

    @Test
    void encodePair_truncation() {
        // "hello world" + "good morning" = 4 tokens + 3 special = 7
        // maxLength=6 should truncate to fit
        EncodedInput result = tokenizer.encode("hello world", "good morning", 6);
        assertThat(result.inputIds().length).isEqualTo(6);
        // Should start with [CLS] and end with [SEP]
        assertThat(result.inputIds()[0]).isEqualTo(2);  // [CLS]
        assertThat(result.inputIds()[result.inputIds().length - 1]).isEqualTo(3);  // [SEP]
    }

    @Test
    void encodePair_truncatesLongerSequenceFirst() {
        // textA = "hello" (1 token), textB = "hello world good" (3 tokens)
        // maxLength = 6, available = 3, lenA=1, lenB=3 -> truncate B to 2
        EncodedInput result = tokenizer.encode("hello", "hello world good", 6);
        // [CLS]=2, hello=5, [SEP]=3, hello=5, world=6, [SEP]=3
        assertThat(result.inputIds()).isEqualTo(new long[]{2, 5, 3, 5, 6, 3});
    }
}
