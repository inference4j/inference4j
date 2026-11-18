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

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import io.github.inference4j.genai.nlp.TextGenerator;
import io.github.inference4j.generation.GenerationResult;

import java.util.ArrayList;
import java.util.List;

/**
 * Converts LangChain4j chat requests and responses for the chat model adapters.
 */
final class ChatRequests {

    private ChatRequests() {
    }

    /**
     * The conversation of {@code request} as inference4j messages.
     *
     * @throws UnsupportedOperationException if the request uses tools, a JSON response format or sampling
     *                                       parameters, or holds a message that is not text
     */
    static List<io.github.inference4j.generation.ChatMessage> messages(ChatRequest request) {
        checkParameters(request.parameters());
        List<io.github.inference4j.generation.ChatMessage> messages = new ArrayList<>();
        for (ChatMessage message : request.messages()) {
            if (message instanceof SystemMessage system) {
                messages.add(io.github.inference4j.generation.ChatMessage.system(system.text()));
            } else if (message instanceof UserMessage user) {
                if (!user.hasSingleText()) {
                    throw new UnsupportedOperationException("Only text user messages are supported");
                }
                messages.add(io.github.inference4j.generation.ChatMessage.user(user.singleText()));
            } else if (message instanceof AiMessage ai) {
                if (ai.hasToolExecutionRequests()) {
                    throw new UnsupportedOperationException("Tool calling is not supported");
                }
                messages.add(io.github.inference4j.generation.ChatMessage.assistant(
                        ai.text() == null ? "" : ai.text()));
            } else {
                throw new UnsupportedOperationException("Unsupported message type: " + message.type());
            }
        }
        return messages;
    }

    /** The response for a generation of {@code generator}. */
    static ChatResponse response(GenerationResult result, TextGenerator generator) {
        boolean newTokenLimit = generator.maxNewTokens() > 0 && result.generatedTokens() >= generator.maxNewTokens();
        boolean lengthLimit = result.promptTokens() + result.generatedTokens() >= generator.maxLength();
        return ChatResponse.builder()
                .aiMessage(AiMessage.from(result.text()))
                .tokenUsage(new TokenUsage(result.promptTokens(), result.generatedTokens()))
                .finishReason(newTokenLimit || lengthLimit ? FinishReason.LENGTH : FinishReason.STOP)
                .build();
    }

    private static void checkParameters(ChatRequestParameters parameters) {
        if (parameters == null) {
            return;
        }
        if (parameters.toolSpecifications() != null && !parameters.toolSpecifications().isEmpty()) {
            throw new UnsupportedOperationException("Tool calling is not supported");
        }
        if (parameters.responseFormat() != null && parameters.responseFormat().type() == ResponseFormatType.JSON) {
            throw new UnsupportedOperationException("JSON response format is not supported");
        }
        List<String> set = new ArrayList<>();
        if (parameters.temperature() != null) {
            set.add("temperature");
        }
        if (parameters.topP() != null) {
            set.add("topP");
        }
        if (parameters.topK() != null) {
            set.add("topK");
        }
        if (parameters.maxOutputTokens() != null) {
            set.add("maxOutputTokens");
        }
        if (parameters.stopSequences() != null && !parameters.stopSequences().isEmpty()) {
            set.add("stopSequences");
        }
        if (parameters.frequencyPenalty() != null) {
            set.add("frequencyPenalty");
        }
        if (parameters.presencePenalty() != null) {
            set.add("presencePenalty");
        }
        if (!set.isEmpty()) {
            throw new UnsupportedOperationException("Per-request " + String.join(", ", set) + " is not supported; "
                    + "set sampling and limits on the TextGenerator builder");
        }
    }
}
