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

import static org.assertj.core.api.Assertions.*;

class MarianTranslatorModelTest {

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class OpusMtEnEs {

        private MarianTranslator translator;

        @BeforeAll
        void setUp() {
            translator = MarianTranslator.builder()
                    .modelId("inference4j/opus-mt-en-es")
                    .maxNewTokens(50)
                    .build();
        }

        @AfterAll
        void tearDown() throws Exception {
            if (translator != null) translator.close();
        }

        @Test
        void translate_producesNonEmptyText() {
            GenerationResult result = translator.translate("Hello, how are you?", token -> {});

            assertThat(result.text().isBlank()).as("Translated text should not be blank").isFalse();
            assertThat(result.generatedTokens() > 0).as("Should generate at least one token").isTrue();
            assertThat(result.duration()).as("Duration should not be null").isNotNull();
        }

        @Test
        void translate_streamsTokensToListener() {
            List<String> streamedTokens = new ArrayList<>();

            GenerationResult result = translator.translate("The weather is nice today",
                    streamedTokens::add);

            assertThat(streamedTokens.isEmpty()).as("Should stream at least one token").isFalse();
            String concatenated = String.join("", streamedTokens);
            assertThat(concatenated).as("Concatenated streamed tokens should match result text").isEqualTo(result.text());
        }

        @Test
        void translate_respectsMaxNewTokens() throws Exception {
            var limited = MarianTranslator.builder()
                    .modelId("inference4j/opus-mt-en-es")
                    .maxNewTokens(5)
                    .build();
            try {
                GenerationResult result = limited.translate(
                        "Good morning, I hope you have a wonderful day", token -> {});

                assertThat(result.generatedTokens() <= 5).as("Should generate at most 5 tokens, got: " + result.generatedTokens()).isTrue();
            } finally {
                limited.close();
            }
        }
    }
}
