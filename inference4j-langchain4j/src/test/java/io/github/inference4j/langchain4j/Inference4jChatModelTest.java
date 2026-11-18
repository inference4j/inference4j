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

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import io.github.inference4j.genai.nlp.TextGenerator;
import io.github.inference4j.generation.ChatMessage;
import io.github.inference4j.generation.GenerationResult;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class Inference4jChatModelTest {

    @Test
    void conversationIsPassedToTheGenerator() {
        TextGenerator generator = generator(new GenerationResult("Hello.", 12, 3, Duration.ZERO));

        ChatResponse response = new Inference4jChatModel(generator).chat(ChatRequest.builder()
                .messages(SystemMessage.from("Be brief."), UserMessage.from("Hi"), AiMessage.from("Hello."),
                        UserMessage.from("Bye"))
                .build());

        verify(generator).chat(List.of(ChatMessage.system("Be brief."), ChatMessage.user("Hi"),
                ChatMessage.assistant("Hello."), ChatMessage.user("Bye")));
        assertThat(response.aiMessage().text()).isEqualTo("Hello.");
        assertThat(response.tokenUsage().inputTokenCount()).isEqualTo(12);
        assertThat(response.tokenUsage().outputTokenCount()).isEqualTo(3);
        assertThat(response.finishReason()).isEqualTo(FinishReason.STOP);
    }

    @Test
    void finishReasonIsLengthWhenALimitIsReached() {
        TextGenerator byNewTokens = generator(new GenerationResult("a", 10, 64, Duration.ZERO));
        when(byNewTokens.maxNewTokens()).thenReturn(64);
        TextGenerator byLength = generator(new GenerationResult("a", 1000, 24, Duration.ZERO));

        assertThat(new Inference4jChatModel(byNewTokens).chat(UserMessage.from("Hi")).finishReason())
                .isEqualTo(FinishReason.LENGTH);
        assertThat(new Inference4jChatModel(byLength).chat(UserMessage.from("Hi")).finishReason())
                .isEqualTo(FinishReason.LENGTH);
    }

    @Test
    void toolsAreRejected() {
        TextGenerator generator = generator(new GenerationResult("a", 1, 1, Duration.ZERO));
        var chatModel = new Inference4jChatModel(generator);

        assertThatThrownBy(() -> chatModel.chat(ChatRequest.builder()
                .messages(UserMessage.from("Hi"))
                .toolSpecifications(ToolSpecification.builder().name("weather").build())
                .build()))
                .isInstanceOf(UnsupportedOperationException.class)
                .hasMessageContaining("Tool calling");
        assertThatThrownBy(() -> chatModel.chat(UserMessage.from("Hi"), AiMessage.from(List.of(
                ToolExecutionRequest.builder().name("weather").arguments("{}").build()))))
                .isInstanceOf(UnsupportedOperationException.class);
        verify(generator, never()).chat(anyList());
    }

    @Test
    void perRequestSamplingIsRejected() {
        var chatModel = new Inference4jChatModel(generator(new GenerationResult("a", 1, 1, Duration.ZERO)));

        assertThatThrownBy(() -> chatModel.chat(ChatRequest.builder()
                .messages(UserMessage.from("Hi"))
                .temperature(0.2)
                .maxOutputTokens(50)
                .build()))
                .isInstanceOf(UnsupportedOperationException.class)
                .hasMessageContaining("temperature, maxOutputTokens")
                .hasMessageContaining("TextGenerator builder");
    }

    @Test
    void jsonResponseFormatAndImagesAreRejected() {
        var chatModel = new Inference4jChatModel(generator(new GenerationResult("a", 1, 1, Duration.ZERO)));

        assertThatThrownBy(() -> chatModel.chat(ChatRequest.builder()
                .messages(UserMessage.from("Hi"))
                .responseFormat(ResponseFormat.JSON)
                .build()))
                .isInstanceOf(UnsupportedOperationException.class);
        assertThatThrownBy(() -> chatModel.chat(UserMessage.from(
                TextContent.from("What is this?"), ImageContent.from("https://example.com/cat.png"))))
                .isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void streamingSendsPartialsThenTheCompleteResponse() {
        TextGenerator generator = mock(TextGenerator.class);
        when(generator.maxLength()).thenReturn(1024);
        when(generator.chat(anyList(), any())).thenAnswer(invocation -> {
            Consumer<String> listener = invocation.getArgument(1);
            listener.accept("Hel");
            listener.accept("lo.");
            return new GenerationResult("Hello.", 5, 2, Duration.ZERO);
        });
        Handler handler = new Handler();

        new Inference4jStreamingChatModel(generator).chat(List.of(UserMessage.from("Hi")), handler);

        assertThat(handler.partials).containsExactly("Hel", "lo.");
        assertThat(handler.complete.aiMessage().text()).isEqualTo("Hello.");
        assertThat(handler.error).isNull();
    }

    @Test
    void streamingReportsUnsupportedRequestsToTheHandler() {
        TextGenerator generator = mock(TextGenerator.class);
        Handler handler = new Handler();

        new Inference4jStreamingChatModel(generator).chat(ChatRequest.builder()
                .messages(UserMessage.from("Hi"))
                .toolSpecifications(ToolSpecification.builder().name("weather").build())
                .build(), handler);

        assertThat(handler.error).isInstanceOf(UnsupportedOperationException.class);
        assertThat(handler.complete).isNull();
        verify(generator, never()).chat(anyList(), any());
    }

    @Test
    void closeClosesTheGenerator() {
        TextGenerator generator = mock(TextGenerator.class);
        TextGenerator streamingGenerator = mock(TextGenerator.class);

        new Inference4jChatModel(generator).close();
        new Inference4jStreamingChatModel(streamingGenerator).close();

        verify(generator).close();
        verify(streamingGenerator).close();
    }

    private static TextGenerator generator(GenerationResult result) {
        TextGenerator generator = mock(TextGenerator.class);
        when(generator.chat(anyList())).thenReturn(result);
        when(generator.maxLength()).thenReturn(1024);
        return generator;
    }

    private static final class Handler implements StreamingChatResponseHandler {

        final List<String> partials = new ArrayList<>();
        ChatResponse complete;
        Throwable error;

        @Override
        public void onPartialResponse(String partialResponse) {
            partials.add(partialResponse);
        }

        @Override
        public void onCompleteResponse(ChatResponse completeResponse) {
            complete = completeResponse;
        }

        @Override
        public void onError(Throwable error) {
            this.error = error;
        }
    }
}
