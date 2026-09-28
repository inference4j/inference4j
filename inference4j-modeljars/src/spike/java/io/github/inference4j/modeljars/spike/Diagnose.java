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

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.runtime.chat.ChatMessage;
import org.modeljars.ModelJar;
import org.modeljars.ModelJars;

import java.util.List;

/** Spike: prints what ModelJars actually does for one generation. */
public final class Diagnose {

    public static void main(String[] args) {
        String coordinate = args.length > 0 ? args[0]
                : "org.modeljars.huggingface:bartowski.deepseek-r1-distill-qwen-1.5b-gguf.q4_k_m:1.0.0-q4_k_m.1";
        try (var runtime = ModelJars.openRuntime(ModelJar.of(coordinate))) {
            System.out.println("template: " + runtime.chatTemplate());
            System.out.println("diagnostics: " + runtime.pipeline().diagnostics());
            System.out.println("execution: " + runtime.executionQualification());
            var prompt = runtime.chatTemplate().render(
                    List.of(ChatMessage.user("Explain what the Java Virtual Machine is in two sentences.")));
            System.out.println("rendered prompt: [" + prompt.text() + "]");
            System.out.println("prompt tokens: " + runtime.pipeline().tokenize(prompt).length);
            var options = SamplingOptions.builder().temperature(0).maxTokens(128).build();
            for (int i = 0; i < 2; i++) {
                try (var session = runtime.openGenerationSession()) {
                    String out = session.generate(prompt, options);
                    System.out.println("metrics: " + session.lastGenerationMetrics());
                    System.out.println("output: " + out.replaceAll("\\s+", " "));
                }
            }
        }
    }

    private Diagnose() {
    }
}
