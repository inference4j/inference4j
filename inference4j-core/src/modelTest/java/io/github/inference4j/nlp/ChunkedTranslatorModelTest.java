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

package io.github.inference4j.nlp;

import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.processing.TruncationPolicy;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.nio.charset.StandardCharsets;
import java.util.Objects;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Translates a 773-word article, about 1,000 tokens, in chunks. The wrapped translators reject over-long
 * input ({@code TruncationPolicy.FAIL}), so a chunk the budget let through over the real limit fails the
 * test.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ChunkedTranslatorModelTest {

    private String article;

    @BeforeAll
    void setUp() throws Exception {
        article = new String(Objects.requireNonNull(getClass().getResourceAsStream("/long-benign-article.txt"))
                .readAllBytes(), StandardCharsets.UTF_8);
    }

    private static int paragraphs(String text) {
        return text.strip().split("\\n\\s*\\n").length;
    }

    private static int words(String text) {
        return text.strip().split("\\s+").length;
    }

    @Test
    void marianTranslatesALongArticleInFull() throws Exception {
        String truncated;
        try (MarianTranslator plain = MarianTranslator.builder().modelId("inference4j/opus-mt-en-es").build()) {
            truncated = plain.translate(article);
        }
        try (ChunkedTranslator translator = new ChunkedTranslator(MarianTranslator.builder()
                .modelId("inference4j/opus-mt-en-es")
                .truncation(TruncationPolicy.FAIL)
                .build())) {
            StringBuilder streamed = new StringBuilder();

            GenerationResult result = translator.translate(article, streamed::append);

            assertThat(result.promptTokens()).isGreaterThan(512);
            assertThat(streamed.toString()).isEqualTo(result.text());
            assertThat(paragraphs(result.text())).isEqualTo(paragraphs(article));
            // The plain translator sees only the first 512 tokens
            assertThat(words(result.text())).isGreaterThan(2 * words(truncated));
            assertThat(words(result.text())).isBetween(words(article) * 8 / 10, words(article) * 14 / 10);
        }
    }

    @Test
    void shortTextIsTranslatedExactlyAsByTheWrappedTranslator() throws Exception {
        String text = "Hello, how are you? The weather is nice today.";
        try (MarianTranslator plain = MarianTranslator.builder().modelId("inference4j/opus-mt-en-es").build();
             ChunkedTranslator chunked = new ChunkedTranslator(
                     MarianTranslator.builder().modelId("inference4j/opus-mt-en-es").build())) {
            assertThat(chunked.translate(text)).isEqualTo(plain.translate(text));
        }
    }

    @Test
    void flanT5CountsItsPromptInTheBudget() throws Exception {
        String excerpt = String.join("\n\n", java.util.Arrays.copyOf(article.split("\n\n"), 3));
        try (ChunkedTranslator translator = new ChunkedTranslator(FlanT5TextGenerator.flanT5Small()
                .maxInputLength(96)
                .truncation(TruncationPolicy.FAIL)
                .build())) {
            GenerationResult result = translator.translate(excerpt, Language.EN, Language.DE, token -> {});

            assertThat(result.promptTokens()).isGreaterThan(96);
            assertThat(paragraphs(result.text())).isEqualTo(3);
        }
    }
}
