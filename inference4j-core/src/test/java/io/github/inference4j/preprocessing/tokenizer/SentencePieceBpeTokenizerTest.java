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
import io.github.inference4j.tokenizer.SentencePieceBpeTokenizer;
import io.github.inference4j.tokenizer.TokenizerJsonParser;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Objects;

import static org.assertj.core.api.Assertions.assertThat;

class SentencePieceBpeTokenizerTest {

    private static SentencePieceBpeTokenizer tokenizer;

    @BeforeAll
    static void setUp() {
        Path tokenizerJson = Path.of(Objects.requireNonNull(
                SentencePieceBpeTokenizerTest.class.getResource("/test-sp-tokenizer.json")).getPath());
        tokenizer = TokenizerJsonParser.parse(tokenizerJson).build();
    }

    @Test
    void encode_prependsSpacePrefix() {
        // "hello" → "▁hello" → should merge to token 14
        EncodedInput result = tokenizer.encode("hello", 512);
        assertThat(result.inputIds()[0]).as("▁hello should be token 14").isEqualTo(14);
        assertThat(result.inputIds().length).as("single merged token").isEqualTo(1);
    }

    @Test
    void encode_handlesSpacesBetweenWords() {
        // "hello world" → "▁hello▁world" → tokens 14, 18
        EncodedInput result = tokenizer.encode("hello world", 512);
        assertThat(result.inputIds()).isEqualTo(new long[]{14, 18});
    }

    @Test
    void encode_addedTokensAtomic() {
        // "<start_of_turn>hello" → special token 300 + ▁hello(14)
        EncodedInput result = tokenizer.encode("<start_of_turn>hello", 512);
        assertThat(result.inputIds()[0]).as("<start_of_turn> should be token 300").isEqualTo(300);
        assertThat(result.inputIds()[1]).as("▁hello should be token 14").isEqualTo(14);
    }

    @Test
    void encode_byteFallbackForUnknownChars() {
        // "!" is not in the vocab, so it should fall back to <0x21> = token 73
        // Input: "!" → "▁!" → ▁ (token 4) + byte fallback for '!'
        EncodedInput result = tokenizer.encode("!", 512);
        // ▁ is token 4, ! (0x21) is token 73
        assertThat(result.inputIds()[0]).as("▁ should be token 4").isEqualTo(4);
        assertThat(result.inputIds()[1]).as("! should be byte fallback <0x21> = 73").isEqualTo(73);
    }

    @Test
    void decode_reversesEncoding() {
        EncodedInput encoded = tokenizer.encode("hello world", 512);
        int[] ids = new int[encoded.inputIds().length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = (int) encoded.inputIds()[i];
        }
        String decoded = tokenizer.decode(ids);
        assertThat(decoded).isEqualTo("hello world");
    }

    @Test
    void decode_singleToken_streaming() {
        // Token 14 is "▁hello" → should decode to "hello" (leading space stripped)
        // But in streaming mode, individual decode returns raw token text with ▁ → space
        String result = tokenizer.decode(14);
        // ▁hello → " hello" (▁ replaced with space)
        assertThat(result).isEqualTo(" hello");
    }

    @Test
    void decode_skipsSpecialTokens() {
        // BOS (2) and EOS (1) are added tokens → should be skipped
        String result = tokenizer.decode(new int[]{2, 14, 18, 1});
        assertThat(result).isEqualTo("hello world");
    }

    @Test
    void decode_byteFallbackTokens() {
        // <0x21> (token 73) is byte 0x21 = '!'
        String result = tokenizer.decode(new int[]{14, 73});
        // ▁hello + ! → "hello!"
        assertThat(result).isEqualTo("hello!");
    }

    @Test
    void encode_attentionMaskAllOnes() {
        EncodedInput result = tokenizer.encode("hello", 512);
        for (long mask : result.attentionMask()) {
            assertThat(mask).isEqualTo(1L);
        }
    }

    @Test
    void encode_tokenTypeIdsAllZeros() {
        EncodedInput result = tokenizer.encode("hello world", 512);
        for (long typeId : result.tokenTypeIds()) {
            assertThat(typeId).isEqualTo(0L);
        }
    }

    @Test
    void encode_truncatesToMaxLength() {
        EncodedInput result = tokenizer.encode("hello world", 1);
        assertThat(result.inputIds().length).isEqualTo(1);
    }

    @Test
    void encode_multipleAddedTokens() {
        // "<start_of_turn>hello<end_of_turn>" → 300, 14, 301
        EncodedInput result = tokenizer.encode("<start_of_turn>hello<end_of_turn>", 512);
        assertThat(result.inputIds()[0]).isEqualTo(300);
        assertThat(result.inputIds().length).isGreaterThanOrEqualTo(3);
        assertThat(result.inputIds()[result.inputIds().length - 1]).isEqualTo(301);
    }

    @Test
    void decode_multiByteFallback() {
        // UTF-8 encoding of 'é' is 0xC3 0xA9
        // <0xC3> = token 235, <0xA9> = token 209
        String result = tokenizer.decode(new int[]{235, 209});
        assertThat(result).as("Two byte-fallback tokens should decode to 'é'").isEqualTo("\u00e9");
    }
}
