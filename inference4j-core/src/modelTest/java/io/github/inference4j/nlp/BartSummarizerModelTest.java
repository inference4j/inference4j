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
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class BartSummarizerModelTest {

    private static final String ARTICLE = "The tower is 324 metres (1,063 ft) tall, about the same height "
            + "as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring "
            + "125 metres (410 ft) on each side. It was the first structure to reach a height of 300 metres.";

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class DistilBartCnn {

        private BartSummarizer summarizer;

        @BeforeAll
        void setUp() {
            summarizer = BartSummarizer.distilBartCnn()
                    .maxNewTokens(30)
                    .build();
        }

        @AfterAll
        void tearDown() throws Exception {
            if (summarizer != null) summarizer.close();
        }

        @Test
        void summarize_producesNonEmptyText() {
            GenerationResult result = summarizer.summarize(ARTICLE, token -> {});

            assertFalse(result.text().isBlank(), "Summary should not be blank");
            assertTrue(result.generatedTokens() > 0, "Should generate at least one token");
            assertNotNull(result.duration(), "Duration should not be null");
        }

        @Test
        void summarize_streamsTokensToListener() {
            List<String> streamedTokens = new ArrayList<>();

            GenerationResult result = summarizer.summarize(ARTICLE, streamedTokens::add);

            assertFalse(streamedTokens.isEmpty(), "Should stream at least one token");
            String concatenated = String.join("", streamedTokens);
            assertEquals(result.text(), concatenated,
                    "Concatenated streamed tokens should match result text");
        }

        @Test
        void summarize_respectsMaxNewTokens() {
            GenerationResult result = summarizer.summarize(ARTICLE, token -> {});

            assertTrue(result.generatedTokens() <= 30,
                    "Should generate at most 30 tokens, got: " + result.generatedTokens());
        }
    }

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class BartLargeCnn {

        private BartSummarizer summarizer;

        @BeforeAll
        void setUp() {
            summarizer = BartSummarizer.bartLargeCnn()
                    .maxNewTokens(30)
                    .build();
        }

        @AfterAll
        void tearDown() throws Exception {
            if (summarizer != null) summarizer.close();
        }

        @Test
        void summarize_producesNonEmptyText() {
            GenerationResult result = summarizer.summarize(ARTICLE, token -> {});

            assertFalse(result.text().isBlank(), "Summary should not be blank");
            assertTrue(result.generatedTokens() > 0, "Should generate at least one token");
        }

        @Test
        void summarize_streamsTokensToListener() {
            List<String> streamedTokens = new ArrayList<>();

            GenerationResult result = summarizer.summarize(ARTICLE, streamedTokens::add);

            assertFalse(streamedTokens.isEmpty(), "Should stream at least one token");
            String concatenated = String.join("", streamedTokens);
            assertEquals(result.text(), concatenated,
                    "Concatenated streamed tokens should match result text");
        }

        @Test
        void summarize_respectsMaxNewTokens() {
            GenerationResult result = summarizer.summarize(ARTICLE, token -> {});

            assertTrue(result.generatedTokens() <= 30,
                    "Should generate at most 30 tokens, got: " + result.generatedTokens());
        }
    }
}
