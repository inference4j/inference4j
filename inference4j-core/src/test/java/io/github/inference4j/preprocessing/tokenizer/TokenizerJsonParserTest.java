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

class TokenizerJsonParserTest {

    private static Path tokenizerJson;

    @BeforeAll
    static void loadPath() {
        tokenizerJson = Path.of(Objects.requireNonNull(
                TokenizerJsonParserTest.class.getResource("/test-sp-tokenizer.json")).getPath());
    }

    @Test
    void parse_returnsNonNullBuilder() {
        SentencePieceBpeTokenizer.Builder builder = TokenizerJsonParser.parse(tokenizerJson);
        assertThat(builder).isNotNull();
    }

    @Test
    void parse_producesWorkingTokenizer() {
        SentencePieceBpeTokenizer tokenizer = TokenizerJsonParser.parse(tokenizerJson).build();

        EncodedInput encoded = tokenizer.encode("hello", 512);
        assertThat(encoded.inputIds().length).as("Should produce at least one token").isGreaterThan(0);
    }

    @Test
    void parse_extractsAddedTokens() {
        SentencePieceBpeTokenizer tokenizer = TokenizerJsonParser.parse(tokenizerJson).build();

        // Added tokens from the fixture include <start_of_turn> (300), <end_of_turn> (301)
        // They should be treated as special tokens (skipped during decode)
        String decoded = tokenizer.decode(new int[]{300, 14, 301});
        assertThat(decoded).as("Special tokens should be skipped during decode").isEqualTo("hello");
    }

    @Test
    void parse_roundTrip() {
        SentencePieceBpeTokenizer tokenizer = TokenizerJsonParser.parse(tokenizerJson).build();

        String input = "hello world";
        EncodedInput encoded = tokenizer.encode(input, 512);
        int[] ids = new int[encoded.inputIds().length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = (int) encoded.inputIds()[i];
        }
        String decoded = tokenizer.decode(ids);
        assertThat(decoded).isEqualTo(input);
    }

    @Test
    void parse_arrayMerges_roundTrip() {
        Path arrayMergesJson = Path.of(Objects.requireNonNull(
                TokenizerJsonParserTest.class.getResource(
                        "/test-sp-tokenizer-array-merges.json")).getPath());
        SentencePieceBpeTokenizer tokenizer =
                TokenizerJsonParser.parse(arrayMergesJson).build();

        String input = "hello world";
        EncodedInput encoded = tokenizer.encode(input, 512);
        int[] ids = new int[encoded.inputIds().length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = (int) encoded.inputIds()[i];
        }
        String decoded = tokenizer.decode(ids);
        assertThat(decoded).isEqualTo(input);
    }
}
