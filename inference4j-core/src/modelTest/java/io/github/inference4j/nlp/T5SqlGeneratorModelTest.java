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

class T5SqlGeneratorModelTest {

    private static final String SCHEMA = "CREATE TABLE users (id INT, name TEXT, email TEXT)";

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class T5SmallAwesome {

        private T5SqlGenerator generator;

        @BeforeAll
        void setUp() {
            generator = T5SqlGenerator.t5SmallAwesome()
                    .maxNewTokens(50)
                    .build();
        }

        @AfterAll
        void tearDown() throws Exception {
            if (generator != null) generator.close();
        }

        @Test
        void generateSql_producesNonEmptyText() {
            GenerationResult result = generator.generateSql("How many users?", SCHEMA, token -> {});

            assertFalse(result.text().isBlank(), "Generated SQL should not be blank");
            assertTrue(result.generatedTokens() > 0, "Should generate at least one token");
            assertNotNull(result.duration(), "Duration should not be null");
        }

        @Test
        void generateSql_streamsTokensToListener() {
            List<String> streamedTokens = new ArrayList<>();

            GenerationResult result = generator.generateSql("List all user names", SCHEMA,
                    streamedTokens::add);

            assertFalse(streamedTokens.isEmpty(), "Should stream at least one token");
            String concatenated = String.join("", streamedTokens);
            assertEquals(result.text(), concatenated,
                    "Concatenated streamed tokens should match result text");
        }

        @Test
        void generateSql_respectsMaxNewTokens() {
            GenerationResult result = generator.generateSql("How many users?", SCHEMA, token -> {});

            assertTrue(result.generatedTokens() <= 50,
                    "Should generate at most 50 tokens, got: " + result.generatedTokens());
        }
    }

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class T5LargeSpider {

        private T5SqlGenerator generator;

        @BeforeAll
        void setUp() {
            generator = T5SqlGenerator.t5LargeSpider()
                    .maxNewTokens(50)
                    .build();
        }

        @AfterAll
        void tearDown() throws Exception {
            if (generator != null) generator.close();
        }

        @Test
        void generateSql_producesNonEmptyText() {
            GenerationResult result = generator.generateSql("How many users?", SCHEMA, token -> {});

            assertFalse(result.text().isBlank(), "Generated SQL should not be blank");
            assertTrue(result.generatedTokens() > 0, "Should generate at least one token");
        }

        @Test
        void generateSql_streamsTokensToListener() {
            List<String> streamedTokens = new ArrayList<>();

            GenerationResult result = generator.generateSql("List all user names", SCHEMA,
                    streamedTokens::add);

            assertFalse(streamedTokens.isEmpty(), "Should stream at least one token");
            String concatenated = String.join("", streamedTokens);
            assertEquals(result.text(), concatenated,
                    "Concatenated streamed tokens should match result text");
        }

        @Test
        void generateSql_respectsMaxNewTokens() {
            GenerationResult result = generator.generateSql("How many users?", SCHEMA, token -> {});

            assertTrue(result.generatedTokens() <= 50,
                    "Should generate at most 50 tokens, got: " + result.generatedTokens());
        }
    }
}
