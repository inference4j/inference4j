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

package io.github.inference4j.langchain4j;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.github.inference4j.genai.nlp.TextGenerator;

import java.util.Objects;

/**
 * A LangChain4j {@link ChatModel} backed by an inference4j {@link TextGenerator}, running the LLM in-process
 * through onnxruntime-genai:
 *
 * <pre>{@code
 * var chatModel = new Inference4jChatModel(TextGenerator.builder()
 *         .model(ModelSources.phi4Mini())
 *         .maxNewTokens(300)
 *         .temperature(0)
 *         .build());
 *
 * Assistant assistant = AiServices.builder(Assistant.class)
 *         .chatModel(chatModel)
 *         .build();
 * }</pre>
 *
 * <p>Needs {@code inference4j-genai} on the classpath. The conversation is formatted with the generator's chat
 * template, so the generator needs one that formats conversations; the models in {@code ModelSources} have one.
 *
 * <p>Sampling and limits are set on the {@link TextGenerator} builder. A request that sets them, uses tools or
 * asks for a JSON response is rejected with {@link UnsupportedOperationException}. The response reports token
 * usage and a finish reason: {@code LENGTH} when the generator's {@code maxNewTokens} or {@code maxLength} was
 * reached, {@code STOP} otherwise.
 *
 * <p>The adapter owns the generator: {@link #close()} closes it. Thread-safe; the generator runs one call at a
 * time.
 *
 * @see Inference4jStreamingChatModel
 */
public class Inference4jChatModel implements ChatModel, AutoCloseable {

    private final TextGenerator generator;

    /**
     * @param generator the generator to answer with; closed by {@link #close()}
     */
    public Inference4jChatModel(TextGenerator generator) {
        this.generator = Objects.requireNonNull(generator, "generator is required");
    }

    @Override
    public ChatResponse doChat(ChatRequest request) {
        return ChatRequests.response(generator.chat(ChatRequests.messages(request)), generator);
    }

    /**
     * Closes the wrapped generator.
     */
    @Override
    public void close() {
        generator.close();
    }
}
