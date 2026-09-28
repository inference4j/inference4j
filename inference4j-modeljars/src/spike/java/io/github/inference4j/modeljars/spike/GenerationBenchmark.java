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

package io.github.inference4j.modeljars.spike;

import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.generation.GenerativeTask;
import io.github.inference4j.genai.ModelSources;
import io.github.inference4j.genai.nlp.TextGenerator;
import io.github.inference4j.modeljars.nlp.ModelJarsTextGenerator;

import java.util.List;
import java.util.function.Supplier;

/**
 * Spike benchmark: the same model (DeepSeek-R1-Distill-Qwen-1.5B, 4-bit) through
 * onnxruntime-genai and through ModelJars, greedy decoding, same prompt and token budget.
 *
 * <p>Args: {@code [genai|modeljars|both] [runs] [maxNewTokens]}; defaults {@code both 3 128}.
 */
public final class GenerationBenchmark {

    private static final String PROMPT = "Explain what the Java Virtual Machine is in two sentences.";

    public static void main(String[] args) throws Exception {
        String which = args.length > 0 ? args[0] : "both";
        int runs = args.length > 1 ? Integer.parseInt(args[1]) : 3;
        int maxNewTokens = args.length > 2 ? Integer.parseInt(args[2]) : 128;

        System.out.printf("Java %s, %d cores, prompt=\"%s\", maxNewTokens=%d, runs=%d%n%n",
                Runtime.version(), Runtime.getRuntime().availableProcessors(), PROMPT, maxNewTokens, runs);

        if (which.equals("genai") || which.equals("both")) {
            bench("onnxruntime-genai (int4 ONNX)", runs, () -> TextGenerator.builder()
                    .model(ModelSources.deepSeekR1_1_5B())
                    // genai's max_length counts prompt tokens too; leave headroom so the
                    // new-token budget is not the limiting factor before the cap below
                    .maxLength(maxNewTokens + 32)
                    .build());
        }
        if (which.equals("modeljars") || which.equals("both")) {
            bench("ModelJars (Q4_K_M GGUF)", runs, () -> ModelJarsTextGenerator.builder()
                    .model("org.modeljars.huggingface:bartowski.deepseek-r1-distill-qwen-1.5b-gguf.q4_k_m:1.0.0-q4_k_m.1")
                    .maxNewTokens(maxNewTokens)
                    .build());
        }
    }

    private static void bench(String name, int runs,
                              Supplier<GenerativeTask<String, GenerationResult>> factory) throws Exception {
        System.out.println("== " + name);
        long loadStart = System.nanoTime();
        try (GenerativeTask<String, GenerationResult> generator = factory.get()) {
            System.out.printf("load: %d ms%n", (System.nanoTime() - loadStart) / 1_000_000);

            // warm-up
            GenerationResult warm = generator.generate(PROMPT);
            System.out.printf("warm-up output (%d tokens): %s%n",
                    warm.generatedTokens(), oneLine(warm.text()));

            for (int i = 0; i < runs; i++) {
                long[] firstToken = {0};
                int[] chunks = {0};
                long start = System.nanoTime();
                GenerationResult result = generator.generate(PROMPT, token -> {
                    if (firstToken[0] == 0) {
                        firstToken[0] = System.nanoTime();
                    }
                    chunks[0]++;
                });
                long end = System.nanoTime();
                double ttftMs = (firstToken[0] - start) / 1e6;
                double decodeSec = (end - firstToken[0]) / 1e9;
                int tokens = result.generatedTokens();
                System.out.printf("run %d: tokens=%d  TTFT=%.0f ms  decode=%.1f tok/s  total=%d ms%n",
                        i + 1, tokens, ttftMs, (tokens - 1) / decodeSec, (end - start) / 1_000_000);
            }
        }
        System.out.println();
    }

    private static String oneLine(String text) {
        String flat = text.replaceAll("\\s+", " ").strip();
        return flat.length() > 160 ? flat.substring(0, 160) + "…" : flat;
    }

    private GenerationBenchmark() {
    }
}
