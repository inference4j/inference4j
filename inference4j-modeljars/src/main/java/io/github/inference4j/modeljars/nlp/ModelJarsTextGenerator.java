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

package io.github.inference4j.modeljars.nlp;

import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.StopReason;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.runtime.TextGenerationSession;
import com.integrallis.models.runtime.chat.ChatMessage;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.generation.GenerationResult;
import io.github.inference4j.generation.GenerativeTask;
import org.modeljars.ModelJar;
import org.modeljars.ModelJarRuntime;
import org.modeljars.ModelJars;

import java.time.Duration;
import java.util.List;
import java.util.function.Consumer;

/**
 * Text generation backed by the <a href="https://modeljars.org">ModelJars</a> runtime.
 *
 * <p>Models are selected by their ModelJars marker — either the generated catalog constant
 * or the full Maven coordinate of a marker JAR on the classpath. ModelJars resolves the
 * pinned artifact, verifies its SHA-256, and picks the qualified backend and chat template;
 * this class adapts that runtime to inference4j's {@link GenerativeTask} contract.
 *
 * <pre>{@code
 * try (var generator = ModelJarsTextGenerator.builder()
 *         .model("org.modeljars.huggingface:qwen.qwen2.5-0.5b-instruct-gguf.q4_k_m:2.5.0-q4_k_m.1")
 *         .maxNewTokens(128)
 *         .build()) {
 *     GenerationResult result = generator.generate("Name one JVM language.");
 * }
 * }</pre>
 *
 * <p>Each call runs in its own generation session, so calls do not share conversation or
 * KV-cache state. Requires Java 25 and {@code --add-modules=jdk.incubator.vector}.
 */
public class ModelJarsTextGenerator implements GenerativeTask<String, GenerationResult> {

    private final ModelJarRuntime runtime;
    private final SamplingOptions samplingOptions;

    private ModelJarsTextGenerator(ModelJarRuntime runtime, SamplingOptions samplingOptions) {
        this.runtime = runtime;
        this.samplingOptions = samplingOptions;
    }

    @Override
    public GenerationResult generate(String input) {
        return generate(input, null);
    }

    @Override
    public GenerationResult generate(String input, Consumer<String> tokenListener) {
        ModelPrompt prompt = runtime.chatTemplate().render(List.of(ChatMessage.user(input)));
        CollectingStream stream = new CollectingStream(tokenListener);

        long start = System.nanoTime();
        try (TextGenerationSession session = runtime.openGenerationSession()) {
            session.generate(prompt, samplingOptions, stream);
        } catch (RuntimeException e) {
            throw new InferenceException("Generation failed: " + e.getMessage(), e);
        }
        Duration duration = Duration.ofNanos(System.nanoTime() - start);

        if (stream.error != null) {
            throw new InferenceException("Generation failed: " + stream.error.getMessage(), stream.error);
        }
        return new GenerationResult(stream.text.toString(),
                stream.usage != null ? stream.usage.promptTokens() : 0,
                stream.usage != null ? stream.usage.completionTokens() : stream.tokenCount,
                duration);
    }

    @Override
    public void close() {
        runtime.close();
    }

    public static Builder builder() {
        return new Builder();
    }

    private static final class CollectingStream implements TokenStream {

        private final Consumer<String> listener;
        private final StringBuilder text = new StringBuilder();
        private int tokenCount;
        private GenerationUsage usage;
        private Throwable error;

        private CollectingStream(Consumer<String> listener) {
            this.listener = listener;
        }

        @Override
        public void onToken(String token) {
            text.append(token);
            tokenCount++;
            if (listener != null && !token.isEmpty()) {
                listener.accept(token);
            }
        }

        @Override
        public void onComplete() {
        }

        @Override
        public void onComplete(GenerationUsage usage, StopReason stopReason) {
            this.usage = usage;
        }

        @Override
        public void onError(Throwable error) {
            this.error = error;
        }
    }

    public static class Builder {

        private ModelJar model;
        private int maxNewTokens = 256;
        private float temperature = 0f;
        private int topK = 0;
        private float topP = 1f;
        private Long seed;

        /**
         * Selects the model by its generated catalog constant, e.g.
         * {@code org.modeljars.catalog.Qwen_Qwen2_5_0_5b_Instruct_Gguf_Q4_K_M.MODEL}.
         */
        public Builder model(ModelJar model) {
            this.model = model;
            return this;
        }

        /**
         * Selects the model by the Maven coordinate of its marker JAR, which must be on the classpath.
         */
        public Builder model(String markerCoordinate) {
            this.model = ModelJar.of(markerCoordinate);
            return this;
        }

        public Builder maxNewTokens(int maxNewTokens) {
            this.maxNewTokens = maxNewTokens;
            return this;
        }

        public Builder temperature(float temperature) {
            this.temperature = temperature;
            return this;
        }

        public Builder topK(int topK) {
            this.topK = topK;
            return this;
        }

        public Builder topP(float topP) {
            this.topP = topP;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public ModelJarsTextGenerator build() {
            if (model == null) {
                throw new IllegalStateException("model is required — pass a ModelJars catalog constant "
                        + "or a marker coordinate");
            }
            SamplingOptions.Builder options = SamplingOptions.builder()
                    .maxTokens(maxNewTokens)
                    .temperature(temperature)
                    .topP(topP);
            // ModelJars rejects topK <= 0; inference4j uses 0 to mean "disabled"
            if (topK > 0) {
                options.topK(topK);
            }
            if (seed != null) {
                options.seed(seed);
            }
            SamplingOptions samplingOptions = options.build();
            ModelJarRuntime runtime;
            try {
                runtime = ModelJars.openRuntime(model);
            } catch (RuntimeException e) {
                throw new ModelLoadException("Failed to open ModelJars runtime for " + model.source()
                        + ": " + e.getMessage(), e);
            }
            return new ModelJarsTextGenerator(runtime, samplingOptions);
        }
    }
}
