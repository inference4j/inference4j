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

class FlanT5TextGeneratorModelTest {

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class FlanT5Small {

        private FlanT5TextGenerator generator;

        @BeforeAll
        void setUp() {
            generator = FlanT5TextGenerator.flanT5Small()
                    .maxNewTokens(30)
                    .build();
        }

        @AfterAll
        void tearDown() throws Exception {
            if (generator != null) generator.close();
        }

        @Test
        void generate_producesNonEmptyText() {
            GenerationResult result = generator.generate("Translate English to German: How are you?");

            assertFalse(result.text().isBlank(), "Generated text should not be blank");
            assertTrue(result.generatedTokens() > 0, "Should generate at least one token");
            assertNotNull(result.duration(), "Duration should not be null");
        }

        @Test
        void generate_streamsTokensToListener() {
            List<String> streamedTokens = new ArrayList<>();

            GenerationResult result = generator.generate("What is the capital of France?",
                    streamedTokens::add);

            assertFalse(streamedTokens.isEmpty(), "Should stream at least one token");
            String concatenated = String.join("", streamedTokens);
            assertEquals(result.text(), concatenated,
                    "Concatenated streamed tokens should match result text");
        }

        @Test
        void generate_respectsMaxNewTokens() {
            GenerationResult result = generator.generate("Summarize: The quick brown fox jumps over the lazy dog.");

            assertTrue(result.generatedTokens() <= 30,
                    "Should generate at most 30 tokens, got: " + result.generatedTokens());
        }
    }

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class FlanT5Base {

        private FlanT5TextGenerator generator;

        @BeforeAll
        void setUp() {
            generator = FlanT5TextGenerator.flanT5Base()
                    .maxNewTokens(30)
                    .build();
        }

        @AfterAll
        void tearDown() throws Exception {
            if (generator != null) generator.close();
        }

        @Test
        void generate_producesNonEmptyText() {
            GenerationResult result = generator.generate("Translate English to German: How are you?");

            assertFalse(result.text().isBlank(), "Generated text should not be blank");
            assertTrue(result.generatedTokens() > 0, "Should generate at least one token");
        }

        @Test
        void generate_streamsTokensToListener() {
            List<String> streamedTokens = new ArrayList<>();

            GenerationResult result = generator.generate("What is the capital of France?",
                    streamedTokens::add);

            assertFalse(streamedTokens.isEmpty(), "Should stream at least one token");
            String concatenated = String.join("", streamedTokens);
            assertEquals(result.text(), concatenated,
                    "Concatenated streamed tokens should match result text");
        }

        @Test
        void generate_respectsMaxNewTokens() {
            GenerationResult result = generator.generate("Summarize: The quick brown fox jumps over the lazy dog.");

            assertTrue(result.generatedTokens() <= 30,
                    "Should generate at most 30 tokens, got: " + result.generatedTokens());
        }
    }

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class FlanT5Large {

        private FlanT5TextGenerator generator;

        @BeforeAll
        void setUp() {
            generator = FlanT5TextGenerator.flanT5Large()
                    .maxNewTokens(30)
                    .build();
        }

        @AfterAll
        void tearDown() throws Exception {
            if (generator != null) generator.close();
        }

        @Test
        void generate_producesNonEmptyText() {
            GenerationResult result = generator.generate("Translate English to German: How are you?");

            assertFalse(result.text().isBlank(), "Generated text should not be blank");
            assertTrue(result.generatedTokens() > 0, "Should generate at least one token");
        }

        @Test
        void generate_streamsTokensToListener() {
            List<String> streamedTokens = new ArrayList<>();

            GenerationResult result = generator.generate("What is the capital of France?",
                    streamedTokens::add);

            assertFalse(streamedTokens.isEmpty(), "Should stream at least one token");
            String concatenated = String.join("", streamedTokens);
            assertEquals(result.text(), concatenated,
                    "Concatenated streamed tokens should match result text");
        }

        @Test
        void generate_respectsMaxNewTokens() {
            GenerationResult result = generator.generate("Summarize: The quick brown fox jumps over the lazy dog.");

            assertTrue(result.generatedTokens() <= 30,
                    "Should generate at most 30 tokens, got: " + result.generatedTokens());
        }
    }
}
