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
 * Summarizes text longer than the model's input limit, by map-reduce and chunk by chunk. The wrapped summarizers reject
 * over-long input ({@code TruncationPolicy.FAIL}), so a chunk or a reduce input the budget let through over
 * the real limit fails the test.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MapReduceSummarizerModelTest {

    private String article;

    @BeforeAll
    void setUp() throws Exception {
        article = new String(Objects.requireNonNull(getClass().getResourceAsStream("/long-benign-article.txt"))
                .readAllBytes(), StandardCharsets.UTF_8);
    }

    @Test
    void distilBartSummarizesTextBeyondItsInputLimit() throws Exception {
        String twoArticles = article + "\n\n" + article;
        try (MapReduceSummarizer summarizer = new MapReduceSummarizer(BartSummarizer.distilBartCnn()
                .maxNewTokens(60)
                .truncation(TruncationPolicy.FAIL)
                .build())) {
            StringBuilder streamed = new StringBuilder();

            GenerationResult result = summarizer.summarize(twoArticles, streamed::append);

            assertThat(result.promptTokens()).isGreaterThan(1024);
            assertThat(result.text()).isNotBlank();
            assertThat(streamed.toString()).isEqualTo(result.text());
        }
    }

    @Test
    void flanT5CountsItsPromptInTheBudget() throws Exception {
        try (MapReduceSummarizer summarizer = new MapReduceSummarizer(FlanT5TextGenerator.flanT5Small()
                .maxNewTokens(60)
                .truncation(TruncationPolicy.FAIL)
                .build())) {
            GenerationResult result = summarizer.summarize(article, token -> {});

            assertThat(result.promptTokens()).isGreaterThan(512);
            assertThat(result.text()).isNotBlank();
        }
    }

    @Test
    void chunkedSummarizerSummarizesEveryChunkOfTextBeyondTheLimit() throws Exception {
        String twoArticles = article + "\n\n" + article;
        try (ChunkedSummarizer summarizer = new ChunkedSummarizer(BartSummarizer.distilBartCnn()
                .maxNewTokens(60)
                .truncation(TruncationPolicy.FAIL)
                .build())) {
            StringBuilder streamed = new StringBuilder();

            GenerationResult result = summarizer.summarize(twoArticles, streamed::append);

            assertThat(result.promptTokens()).isGreaterThan(1024);
            assertThat(result.text().split("\n\n")).hasSizeGreaterThanOrEqualTo(2)
                    .allSatisfy(summary -> assertThat(summary).isNotBlank().doesNotStartWith(" "));
            assertThat(streamed.toString()).isEqualTo(result.text());
        }
    }

    @Test
    void shortTextIsSummarizedExactlyAsByTheWrappedSummarizer() throws Exception {
        String paragraph = article.strip().split("\\n\\s*\\n")[0];
        String plain;
        try (BartSummarizer summarizer = BartSummarizer.distilBartCnn().maxNewTokens(60).build()) {
            plain = summarizer.summarize(paragraph);
        }
        try (MapReduceSummarizer summarizer = new MapReduceSummarizer(
                BartSummarizer.distilBartCnn().maxNewTokens(60).build())) {
            assertThat(summarizer.summarize(paragraph)).isEqualTo(plain);
        }
    }
}
