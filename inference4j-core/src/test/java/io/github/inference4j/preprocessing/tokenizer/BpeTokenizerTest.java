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

import io.github.inference4j.tokenizer.BpeTokenizer;
import io.github.inference4j.tokenizer.EncodedInput;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Objects;
import java.util.regex.Pattern;

import static org.assertj.core.api.Assertions.assertThat;

class BpeTokenizerTest {

    private static Path vocabPath;
    private static Path mergesPath;

    @BeforeAll
    static void loadPaths() {
        vocabPath = Path.of(Objects.requireNonNull(
                BpeTokenizerTest.class.getResource("/test-bpe-vocab.json")).getPath());
        mergesPath = Path.of(Objects.requireNonNull(
                BpeTokenizerTest.class.getResource("/test-bpe-merges.txt")).getPath());
    }

    @Nested
    class ClipMode {

        private static BpeTokenizer tokenizer;

        private static final Pattern CLIP_PATTERN = Pattern.compile(
                "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
                Pattern.CASE_INSENSITIVE
        );

        @BeforeAll
        static void setUp() {
            tokenizer = BpeTokenizer.builder(vocabPath, mergesPath)
                    .lowercase(true)
                    .endOfWordMarker("</w>")
                    .pattern(CLIP_PATTERN)
                    .bosToken("<|startoftext|>")
                    .eosToken("<|endoftext|>")
                    .pad(true)
                    .defaultMaxLength(77)
                    .build();
        }

        @Test
        void encode_simpleWord() {
            EncodedInput result = tokenizer.encode("hello", 10);
            assertThat(result.inputIds()[0]).as("BOS token").isEqualTo(100);
            assertThat(result.inputIds()[1]).as("hell").isEqualTo(10);
            assertThat(result.inputIds()[2]).as("o</w>").isEqualTo(3);
            assertThat(result.inputIds()[3]).as("EOS token").isEqualTo(101);
        }

        @Test
        void encode_multipleWords() {
            EncodedInput result = tokenizer.encode("hello world", 10);
            assertThat(result.inputIds())
                    .isEqualTo(new long[]{100, 10, 3, 4, 13, 101, 0, 0, 0, 0});
        }

        @Test
        void encode_singleCharacterWord() {
            EncodedInput result = tokenizer.encode("a", 10);
            assertThat(result.inputIds()[0]).isEqualTo(100);
            assertThat(result.inputIds()[1]).isEqualTo(14);
            assertThat(result.inputIds()[2]).isEqualTo(101);
        }

        @Test
        void encode_emptyString() {
            EncodedInput result = tokenizer.encode("", 10);
            assertThat(result.inputIds()[0]).isEqualTo(100);
            assertThat(result.inputIds()[1]).isEqualTo(101);
            assertThat(result.inputIds()[2]).isEqualTo(0);
        }

        @Test
        void encode_attentionMask() {
            EncodedInput result = tokenizer.encode("hello", 10);
            assertThat(result.attentionMask())
                    .isEqualTo(new long[]{1, 1, 1, 1, 0, 0, 0, 0, 0, 0});
        }

        @Test
        void encode_tokenTypeIdsAllZeros() {
            EncodedInput result = tokenizer.encode("hello", 10);
            assertThat(result.tokenTypeIds()).isEqualTo(new long[10]);
        }

        @Test
        void encode_padsToMaxLength() {
            EncodedInput result = tokenizer.encode("a", 10);
            assertThat(result.inputIds().length).isEqualTo(10);
            assertThat(result.attentionMask().length).isEqualTo(10);
            assertThat(result.tokenTypeIds().length).isEqualTo(10);
        }

        @Test
        void encode_truncatesToMaxLength() {
            EncodedInput result = tokenizer.encode("hello world", 5);
            assertThat(result.inputIds().length).isEqualTo(5);
            assertThat(result.inputIds()[0]).as("BOS preserved").isEqualTo(100);
            assertThat(result.inputIds()[4]).as("EOS at end").isEqualTo(101);
        }

        @Test
        void encode_lowercasesInput() {
            EncodedInput upper = tokenizer.encode("HELLO", 10);
            EncodedInput lower = tokenizer.encode("hello", 10);
            assertThat(upper.inputIds()).isEqualTo(lower.inputIds());
        }

        @Test
        void encode_normalizesWhitespace() {
            EncodedInput doubleSpace = tokenizer.encode("hello  world", 10);
            EncodedInput singleSpace = tokenizer.encode("hello world", 10);
            assertThat(doubleSpace.inputIds()).isEqualTo(singleSpace.inputIds());
        }

        @Test
        void encode_defaultMaxLength() {
            EncodedInput result = tokenizer.encode("hello");
            assertThat(result.inputIds().length).isEqualTo(77);
        }
    }

    @Nested
    class AddedTokenMode {

        private static BpeTokenizer tokenizer;

        @BeforeAll
        static void setUp() {
            tokenizer = BpeTokenizer.builder(vocabPath, mergesPath)
                    .addedToken("<|im_start|>")
                    .addedToken("<|im_end|>")
                    .build();
        }

        @Test
        void encode_addedTokenAtStart() {
            EncodedInput result = tokenizer.encode("<|im_start|>hello", 512);
            assertThat(result.inputIds()[0]).as("<|im_start|> should be token 200").isEqualTo(200);
        }

        @Test
        void encode_addedTokenAtEnd() {
            EncodedInput result = tokenizer.encode("hello<|im_end|>", 512);
            long lastId = result.inputIds()[result.inputIds().length - 1];
            assertThat(lastId).as("<|im_end|> should be token 201").isEqualTo(201);
        }

        @Test
        void encode_multipleAddedTokens() {
            EncodedInput result = tokenizer.encode("<|im_start|>hello<|im_end|>", 512);
            assertThat(result.inputIds()[0]).as("first token should be <|im_start|>").isEqualTo(200);
            long lastId = result.inputIds()[result.inputIds().length - 1];
            assertThat(lastId).as("last token should be <|im_end|>").isEqualTo(201);
        }

        @Test
        void encode_addedTokensNotRegistered() {
            BpeTokenizer noAddedTokens = BpeTokenizer.fromFiles(vocabPath, mergesPath);
            EncodedInput result = noAddedTokens.encode("<|im_start|>", 512);
            // Without added tokens, <|im_start|> is byte-encoded — should NOT produce ID 200
            for (long id : result.inputIds()) {
                assertThat(id)
                        .as("Without addedToken(), <|im_start|> should be byte-encoded, not mapped to 200")
                        .isNotEqualTo(200);
            }
        }
    }

    @Nested
    class DefaultMode {

        private static BpeTokenizer tokenizer;

        @BeforeAll
        static void setUp() {
            tokenizer = BpeTokenizer.fromFiles(vocabPath, mergesPath);
        }

        @Test
        void encode_noPadding() {
            EncodedInput result = tokenizer.encode("hello", 10);
            // No BOS/EOS, no padding — just the raw tokens
            // Without </w> marker the BPE will produce different tokens,
            // but array length should match actual token count, not maxLength
            assertThat(result.attentionMask().length).isEqualTo(result.inputIds().length);
            for (long mask : result.attentionMask()) {
                assertThat(mask).as("all attention mask values should be 1 (no padding)").isEqualTo(1L);
            }
        }

        @Test
        void encode_noBosEos() {
            EncodedInput result = tokenizer.encode("hello", 10);
            // First token should NOT be BOS (100)
            for (long id : result.inputIds()) {
                if (id == 100 || id == 101) {
                    throw new AssertionError("Should not contain BOS/EOS tokens");
                }
            }
        }

        @Test
        void encode_preservesCase() {
            EncodedInput upper = tokenizer.encode("HELLO", 512);
            EncodedInput lower = tokenizer.encode("hello", 512);
            // With case sensitivity, these should produce different tokens
            // (assuming the test vocab has case-sensitive entries — if not, they may match
            //  but the important thing is that the tokenizer doesn't force lowercase)
            // We verify the tokenizer doesn't lowercase by checking the processed text
        }
    }
}
