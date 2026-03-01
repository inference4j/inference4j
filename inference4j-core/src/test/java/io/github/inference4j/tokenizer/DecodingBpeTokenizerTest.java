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

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.*;

class DecodingBpeTokenizerTest {

    private DecodingBpeTokenizer createTokenizer() {
        return createTokenizer(null, null);
    }

    private DecodingBpeTokenizer createTokenizer(String bosToken, String eosToken) {
        Path vocabPath = resourcePath("/test-bpe-vocab.json");
        Path mergesPath = resourcePath("/test-bpe-merges.txt");

        BpeTokenizer.Builder builder = BpeTokenizer.builder(vocabPath, mergesPath)
                .endOfWordMarker("</w>");

        if (bosToken != null) {
            builder.bosToken(bosToken);
        }
        if (eosToken != null) {
            builder.eosToken(eosToken);
        }

        return DecodingBpeTokenizer.from(builder);
    }

    private Path resourcePath(String name) {
        try {
            return Paths.get(getClass().getResource(name).toURI());
        }
        catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void decode_singleToken() {
        DecodingBpeTokenizer tokenizer = createTokenizer();

        String result = tokenizer.decode(new int[]{0});

        assertThat(result).isEqualTo("h");
    }

    @Test
    void decode_multipleTokens() {
        DecodingBpeTokenizer tokenizer = createTokenizer();

        // Token 10 = "hell", token 3 = "o</w>"
        // Concatenated BPE string: "hello</w>"
        // reverseBytePairEncoding maps ASCII chars to themselves
        String result = tokenizer.decode(new int[]{10, 3});

        assertThat(result).isEqualTo("hello</w>");
    }

    @Test
    void decode_skipsSpecialTokens() {
        DecodingBpeTokenizer tokenizer = createTokenizer(
                "<|startoftext|>", "<|endoftext|>");

        // Tokens: BOS (100), "hell" (10), "o</w>" (3), EOS (101)
        // BOS and EOS should be excluded from output
        String result = tokenizer.decode(new int[]{100, 10, 3, 101});

        assertThat(result).isEqualTo("hello</w>");
    }

    @Test
    void decode_unknownTokenId_skipped() {
        DecodingBpeTokenizer tokenizer = createTokenizer();

        String result = tokenizer.decode(new int[]{9999});

        assertThat(result).isEmpty();
    }

    @Test
    void decodeSingle_validToken() {
        DecodingBpeTokenizer tokenizer = createTokenizer();

        String result = tokenizer.decode(0);

        assertThat(result).isEqualTo("h");
    }

    @Test
    void decodeSingle_specialToken_returnsEmpty() {
        DecodingBpeTokenizer tokenizer = createTokenizer(
                "<|startoftext|>", "<|endoftext|>");

        String result = tokenizer.decode(100);

        assertThat(result).isEmpty();
    }

    @Test
    void provider_returnsValidProvider() {
        TokenizerProvider provider = DecodingBpeTokenizer.provider();

        assertThat(provider).isNotNull();
        assertThat(provider.requiredFiles()).containsExactly("vocab.json", "merges.txt");
    }

    @Test
    void fromFiles_createsWorkingTokenizer() {
        Path vocabPath = resourcePath("/test-bpe-vocab.json");
        Path mergesPath = resourcePath("/test-bpe-merges.txt");

        DecodingBpeTokenizer tokenizer = DecodingBpeTokenizer.fromFiles(vocabPath, mergesPath);

        EncodedInput encoded = tokenizer.encode("hello");
        assertThat(encoded.inputIds()).isNotEmpty();
    }
}
