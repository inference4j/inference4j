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
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class OnnxTextGeneratorModelTest {

    @Nested
    class Gpt2 {

        @Test
        void generate_producesNonEmptyText() {
            try (var gen = OnnxTextGenerator.gpt2().maxNewTokens(20).build()) {
                GenerationResult result = gen.generate("Once upon a time");

                assertThat(result.text().isBlank()).as("Generated text should not be blank").isFalse();
                assertThat(result.generatedTokens() > 0).as("Should generate at least one token").isTrue();
                assertThat(result.duration()).as("Duration should not be null").isNotNull();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_respectsMaxNewTokens() {
            try (var gen = OnnxTextGenerator.gpt2().maxNewTokens(5).build()) {
                GenerationResult result = gen.generate("The meaning of life is");

                assertThat(result.generatedTokens() <= 5).as("Should generate at most 5 tokens, got: " + result.generatedTokens()).isTrue();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_streamsTokensToListener() {
            try (var gen = OnnxTextGenerator.gpt2().maxNewTokens(20).build()) {
                List<String> streamedTokens = new ArrayList<>();

                GenerationResult result = gen.generate("The quick brown fox", streamedTokens::add);

                assertThat(streamedTokens.isEmpty()).as("Should stream at least one token").isFalse();
                String concatenated = String.join("", streamedTokens);
                assertThat(concatenated).as("Concatenated streamed tokens should match result text").isEqualTo(result.text());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Nested
    class SmolLM2 {

        @Test
        void generate_producesNonEmptyText() {
            try (var gen = OnnxTextGenerator.smolLM2().maxNewTokens(20).build()) {
                GenerationResult result = gen.generate("What is the capital of France?");

                assertThat(result.text().isBlank()).as("Generated text should not be blank").isFalse();
                assertThat(result.generatedTokens() > 0).as("Should generate at least one token").isTrue();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_respectsMaxNewTokens() {
            try (var gen = OnnxTextGenerator.smolLM2().maxNewTokens(5).build()) {
                GenerationResult result = gen.generate("Tell me a joke");

                assertThat(result.generatedTokens() <= 5).as("Should generate at most 5 tokens, got: " + result.generatedTokens()).isTrue();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_streamsTokensToListener() {
            try (var gen = OnnxTextGenerator.smolLM2().maxNewTokens(20).build()) {
                List<String> streamedTokens = new ArrayList<>();

                GenerationResult result = gen.generate("Explain gravity", streamedTokens::add);

                assertThat(streamedTokens.isEmpty()).as("Should stream at least one token").isFalse();
                String concatenated = String.join("", streamedTokens);
                assertThat(concatenated).as("Concatenated streamed tokens should match result text").isEqualTo(result.text());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Nested
    class Qwen2 {

        @Test
        void generate_producesNonEmptyText() {
            try (var gen = OnnxTextGenerator.qwen2().maxNewTokens(20).build()) {
                GenerationResult result = gen.generate("What is 2+2?");

                assertThat(result.text().isBlank()).as("Generated text should not be blank").isFalse();
                assertThat(result.generatedTokens() > 0).as("Should generate at least one token").isTrue();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_respectsMaxNewTokens() {
            try (var gen = OnnxTextGenerator.qwen2().maxNewTokens(5).build()) {
                GenerationResult result = gen.generate("Explain gravity");

                assertThat(result.generatedTokens() <= 5).as("Should generate at most 5 tokens, got: " + result.generatedTokens()).isTrue();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_streamsTokensToListener() {
            try (var gen = OnnxTextGenerator.qwen2().maxNewTokens(20).build()) {
                List<String> streamedTokens = new ArrayList<>();

                GenerationResult result = gen.generate("What is the capital of France?",
                        streamedTokens::add);

                assertThat(streamedTokens.isEmpty()).as("Should stream at least one token").isFalse();
                String concatenated = String.join("", streamedTokens);
                assertThat(concatenated).as("Concatenated streamed tokens should match result text").isEqualTo(result.text());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Nested
    class TinyLlama {

        @Test
        void generate_producesNonEmptyText() {
            try (var gen = OnnxTextGenerator.tinyLlama().maxNewTokens(20).build()) {
                GenerationResult result = gen.generate("What is the capital of France?");

                assertThat(result.text().isBlank()).as("Generated text should not be blank").isFalse();
                assertThat(result.generatedTokens() > 0).as("Should generate at least one token").isTrue();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_respectsMaxNewTokens() {
            try (var gen = OnnxTextGenerator.tinyLlama().maxNewTokens(5).build()) {
                GenerationResult result = gen.generate("Tell me a joke");

                assertThat(result.generatedTokens() <= 5).as("Should generate at most 5 tokens, got: " + result.generatedTokens()).isTrue();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_streamsTokensToListener() {
            try (var gen = OnnxTextGenerator.tinyLlama().maxNewTokens(20).build()) {
                List<String> streamedTokens = new ArrayList<>();

                GenerationResult result = gen.generate("Explain gravity", streamedTokens::add);

                assertThat(streamedTokens.isEmpty()).as("Should stream at least one token").isFalse();
                String concatenated = String.join("", streamedTokens);
                assertThat(concatenated).as("Concatenated streamed tokens should match result text").isEqualTo(result.text());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Nested
    class Gemma2 {

        private static final Path GEMMA_MODEL_DIR = Path.of(
                System.getProperty("user.home"), ".cache", "inference4j", "gemma-2-2b-it");

        @Test
        void generate_producesNonEmptyText() {
            assumeTrue(Files.isDirectory(GEMMA_MODEL_DIR),
                    "Gated model — requires manual download to " + GEMMA_MODEL_DIR);

            try (var gen = OnnxTextGenerator.gemma2()
                    .modelSource(id -> GEMMA_MODEL_DIR)
                    .maxNewTokens(20).build()) {
                GenerationResult result = gen.generate("What is the capital of France?");

                assertThat(result.text().isBlank()).as("Generated text should not be blank").isFalse();
                assertThat(result.generatedTokens() > 0).as("Should generate at least one token").isTrue();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_respectsMaxNewTokens() {
            assumeTrue(Files.isDirectory(GEMMA_MODEL_DIR),
                    "Gated model — requires manual download to " + GEMMA_MODEL_DIR);

            try (var gen = OnnxTextGenerator.gemma2()
                    .modelSource(id -> GEMMA_MODEL_DIR)
                    .maxNewTokens(5).build()) {
                GenerationResult result = gen.generate("Explain gravity");

                assertThat(result.generatedTokens() <= 5).as("Should generate at most 5 tokens, got: " + result.generatedTokens()).isTrue();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Test
        void generate_streamsTokensToListener() {
            assumeTrue(Files.isDirectory(GEMMA_MODEL_DIR),
                    "Gated model — requires manual download to " + GEMMA_MODEL_DIR);

            try (var gen = OnnxTextGenerator.gemma2()
                    .modelSource(id -> GEMMA_MODEL_DIR)
                    .maxNewTokens(20).build()) {
                List<String> streamedTokens = new ArrayList<>();

                GenerationResult result = gen.generate("What is 2+2?",
                        streamedTokens::add);

                assertThat(streamedTokens.isEmpty()).as("Should stream at least one token").isFalse();
                String concatenated = String.join("", streamedTokens);
                assertThat(concatenated).as("Concatenated streamed tokens should match result text").isEqualTo(result.text());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
