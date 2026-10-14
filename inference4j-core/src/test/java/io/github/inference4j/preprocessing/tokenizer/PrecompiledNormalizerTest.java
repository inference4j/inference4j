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

import io.github.inference4j.tokenizer.PrecompiledNormalizer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Objects;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Expected values are HuggingFace {@code tokenizers}' output for SentencePiece's {@code nmt_nfkc} charsmap,
 * the one shipped by T5 and DeBERTa-v3.
 */
class PrecompiledNormalizerTest {

    private static PrecompiledNormalizer normalizer;

    @BeforeAll
    static void setUp() throws Exception {
        Path charsmap = Path.of(Objects.requireNonNull(
                PrecompiledNormalizerTest.class.getResource("/test-nmt-nfkc-charsmap.txt")).toURI());
        normalizer = PrecompiledNormalizer.fromBase64(Files.readString(charsmap).strip());
    }

    @ParameterizedTest
    @CsvSource(delimiter = '|', value = {
            "Hello, world!|Hello, world!",
            "ＡＢＣ１２３|ABC123",
            "ﬁle|file",
            "½|1⁄2",
            "™|TM",
            "㎒|MHz",
            "ﾊﾝ|ハン",
            "é|é",
            "café 😀|café 😀",
    })
    void matchesTheReferenceNormalization(String input, String expected) {
        assertThat(normalizer.apply(input)).isEqualTo(expected);
    }

    @Test
    void controlAndInvisibleWhitespaceBecomesASpace() {
        assertThat(normalizer.apply("a\tb")).isEqualTo("a b");
        assertThat(normalizer.apply("a\r\nb")).isEqualTo("a b");
        assertThat(normalizer.apply("x​y")).isEqualTo("x y");
    }

    @Test
    void repeatedSpacesAreLeftForOtherNormalizers() {
        assertThat(normalizer.apply("a    b")).isEqualTo("a    b");
    }

    @Test
    void emptyTextStaysEmpty() {
        assertThat(normalizer.apply("")).isEmpty();
    }

    @Test
    void rejectsACharsmapWithAnInvalidTrieSize() {
        String broken = Base64.getEncoder().encodeToString(new byte[]{(byte) 0xFF, 0, 0, 0});

        assertThatThrownBy(() -> PrecompiledNormalizer.fromBase64(broken))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
