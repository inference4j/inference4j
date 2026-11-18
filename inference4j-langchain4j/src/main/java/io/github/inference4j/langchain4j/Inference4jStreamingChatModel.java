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

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import io.github.inference4j.genai.nlp.TextGenerator;
import io.github.inference4j.generation.GenerationResult;

import java.util.List;
import java.util.Objects;

/**
 * A LangChain4j {@link StreamingChatModel} backed by an inference4j {@link TextGenerator}, running the LLM
 * in-process through onnxruntime-genai and streaming the answer as it is generated:
 *
 * <pre>{@code
 * var chatModel = new Inference4jStreamingChatModel(TextGenerator.builder()
 *         .model(ModelSources.phi4Mini())
 *         .maxNewTokens(300)
 *         .build());
 *
 * Assistant assistant = AiServices.builder(Assistant.class)
 *         .streamingChatModel(chatModel)
 *         .build();
 * }</pre>
 *
 * <p>Generation runs on the calling thread: the handler receives every partial response, then the complete
 * response or the error, before {@code chat} returns. Requests are handled as in {@link Inference4jChatModel};
 * an unsupported request is reported to {@link StreamingChatResponseHandler#onError(Throwable)}.
 *
 * <p>The adapter owns the generator: {@link #close()} closes it. Thread-safe; the generator runs one call at a
 * time.
 *
 * @see Inference4jChatModel
 */
public class Inference4jStreamingChatModel implements StreamingChatModel, AutoCloseable {

    private final TextGenerator generator;

    /**
     * @param generator the generator to answer with; closed by {@link #close()}
     */
    public Inference4jStreamingChatModel(TextGenerator generator) {
        this.generator = Objects.requireNonNull(generator, "generator is required");
    }

    @Override
    public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        GenerationResult result;
        try {
            List<io.github.inference4j.generation.ChatMessage> messages = ChatRequests.messages(request);
            result = generator.chat(messages, handler::onPartialResponse);
        } catch (RuntimeException e) {
            handler.onError(e);
            return;
        }
        handler.onCompleteResponse(ChatRequests.response(result, generator));
    }

    /**
     * Closes the wrapped generator.
     */
    @Override
    public void close() {
        generator.close();
    }
}
