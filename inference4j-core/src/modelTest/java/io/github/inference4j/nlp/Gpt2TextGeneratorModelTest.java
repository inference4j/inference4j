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
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class Gpt2TextGeneratorModelTest {

    @Test
    void generate_producesNonEmptyText() {
        try (var gen = Gpt2TextGenerator.builder().maxNewTokens(20).build()) {
            GenerationResult result = gen.generate("Once upon a time");

            assertFalse(result.text().isBlank(), "Generated text should not be blank");
            assertTrue(result.generatedTokens() > 0, "Should generate at least one token");
            assertNotNull(result.duration(), "Duration should not be null");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void generate_respectsMaxNewTokens() {
        try (var gen = Gpt2TextGenerator.builder().maxNewTokens(5).build()) {
            GenerationResult result = gen.generate("The meaning of life is");

            assertTrue(result.generatedTokens() <= 5,
                    "Should generate at most 5 tokens, got: " + result.generatedTokens());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void generate_streamsTokensToListener() {
        try (var gen = Gpt2TextGenerator.builder().maxNewTokens(20).build()) {
            List<String> streamedTokens = new ArrayList<>();

            GenerationResult result = gen.generate("The quick brown fox", streamedTokens::add);

            assertFalse(streamedTokens.isEmpty(), "Should stream at least one token");
            String concatenated = String.join("", streamedTokens);
            assertEquals(result.text(), concatenated,
                    "Concatenated streamed tokens should match result text");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
