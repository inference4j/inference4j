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

import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.guardrail.InputGuardrail;
import dev.langchain4j.guardrail.InputGuardrailRequest;
import dev.langchain4j.guardrail.InputGuardrailResult;
import dev.langchain4j.invocation.InvocationContext;
import io.github.inference4j.nlp.InjectionCheck;
import io.github.inference4j.nlp.PromptInjectionDetector;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * A LangChain4j {@link InputGuardrail} that blocks prompt injection and jailbreak attempts with a local
 * classifier, before the message reaches the language model.
 *
 * <pre>{@code
 * var guardrail = new Inference4jPromptInjectionGuardrail(DebertaPromptInjectionDetector.builder().build());
 *
 * Assistant assistant = AiServices.builder(Assistant.class)
 *         .chatModel(chatModel)
 *         .inputGuardrails(guardrail)
 *         .build();
 * }</pre>
 *
 * <p>LangChain4j's {@code PatternBasedPromptInjectionGuardrail} matches known phrasings with regular expressions
 * and suggests chaining a classifier for novel or obfuscated attacks. This is that classifier, in-process, with no
 * extra LLM call per message. Put the regex guardrail first and this one second.
 *
 * <p>An injection fails the message with {@code fatal}, so later guardrails and the model are not called and
 * the AI service throws {@code InputGuardrailException}. The failure message gives the probability and threshold,
 * never the user's text. The threshold is the detector's: set it with
 * {@code DebertaPromptInjectionDetector.builder().threshold(...)}.
 *
 * <p>Only what the user asked is checked. With retrieval-augmented generation, LangChain4j hands guardrails the
 * message with the retrieved content appended; this guardrail checks the original message instead
 * ({@code InvocationContext.originalUserMessage()}), so retrieved documents cannot block a turn. Text parts are
 * checked together; images and other non-text parts are ignored; a message without text passes. Detector errors
 * propagate rather than letting the message through.
 *
 * <p>LangChain4j creates guardrails named in {@code @InputGuardrails(...)} through a no-argument constructor. This
 * guardrail needs a model, so register an instance with {@code AiServices.builder(...).inputGuardrails(...)} or as
 * a bean.
 *
 * <p>The guardrail owns the detector: {@link #close()} closes it. Thread-safe: one instance can guard a whole
 * application.
 */
public class Inference4jPromptInjectionGuardrail implements InputGuardrail, AutoCloseable {

    private final PromptInjectionDetector detector;

    /**
     * @param detector the classifier to check messages with; closed by {@link #close()}
     */
    public Inference4jPromptInjectionGuardrail(PromptInjectionDetector detector) {
        this.detector = Objects.requireNonNull(detector, "detector is required");
    }

    /**
     * Checks the original user message when the request carries one, otherwise the message given.
     */
    @Override
    public InputGuardrailResult validate(InputGuardrailRequest request) {
        return validate(originalMessage(request));
    }

    /**
     * Fails the message with {@code fatal} when the detector flags its text as an injection.
     */
    @Override
    public InputGuardrailResult validate(UserMessage userMessage) {
        String text = text(userMessage);
        if (text.isBlank()) {
            return success();
        }
        InjectionCheck check = detector.check(text);
        if (check.isInjection()) {
            return fatal(String.format(Locale.ROOT, "Prompt injection detected (probability %.2f, threshold %.2f)",
                    check.probability(), check.threshold()));
        }
        return success();
    }

    /**
     * Runs {@link #validate(InputGuardrailRequest)} on the calling thread; the check does no blocking I/O.
     */
    @Override
    public CompletableFuture<InputGuardrailResult> validateAsync(InputGuardrailRequest request) {
        try {
            return CompletableFuture.completedFuture(validate(request));
        } catch (RuntimeException e) {
            return CompletableFuture.failedFuture(e);
        }
    }

    /**
     * Closes the detector.
     */
    @Override
    public void close() {
        detector.close();
    }

    private static UserMessage originalMessage(InputGuardrailRequest request) {
        if (request.requestParams() != null) {
            InvocationContext context = request.requestParams().invocationContext();
            if (context != null && context.originalUserMessage() != null) {
                return context.originalUserMessage();
            }
        }
        return request.userMessage();
    }

    private static String text(UserMessage message) {
        if (message == null) {
            return "";
        }
        List<String> parts = new ArrayList<>();
        for (Content content : message.contents()) {
            if (content instanceof TextContent textContent) {
                parts.add(textContent.text());
            }
        }
        return String.join("\n", parts);
    }
}
