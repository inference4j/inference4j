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

package io.github.inference4j.genai;

import io.github.inference4j.generation.ChatMessage;
import io.github.inference4j.generation.ChatMessage.Role;
import io.github.inference4j.generation.ChatTemplate;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;

/**
 * Chat templates of the models in {@link ModelSources}. Each formats a single user message and a whole
 * conversation; a single message is formatted as a conversation of one user message.
 */
final class PresetChatTemplates {

    private PresetChatTemplates() {
    }

    /** Phi-3: {@code <|role|>} markers, each followed by a newline. */
    static ChatTemplate phi3() {
        return of(messages -> {
            StringBuilder prompt = new StringBuilder();
            for (ChatMessage message : messages) {
                prompt.append("<|").append(role(message)).append("|>\n").append(message.content()).append("<|end|>\n");
            }
            return prompt.append("<|assistant|>\n").toString();
        });
    }

    /** Phi-4: {@code <|role|>} markers without newlines. */
    static ChatTemplate phi4() {
        return of(messages -> {
            StringBuilder prompt = new StringBuilder();
            for (ChatMessage message : messages) {
                prompt.append("<|").append(role(message)).append("|>").append(message.content()).append("<|end|>");
            }
            return prompt.append("<|assistant|>").toString();
        });
    }

    /**
     * DeepSeek-R1: the system prompt (the last system message) after the beginning-of-sentence token, then
     * {@code <｜User｜>} and {@code <｜Assistant｜>} turns. Earlier answers drop their reasoning, as the original
     * template does.
     */
    static ChatTemplate deepSeekR1() {
        return of(messages -> {
            String system = "";
            for (ChatMessage message : messages) {
                if (message.role() == Role.SYSTEM) {
                    system = message.content();
                }
            }
            StringBuilder prompt = new StringBuilder("<｜begin▁of▁sentence｜>").append(system);
            for (ChatMessage message : messages) {
                if (message.role() == Role.USER) {
                    prompt.append("<｜User｜>").append(message.content());
                }
                else if (message.role() == Role.ASSISTANT) {
                    String content = message.content();
                    int think = content.lastIndexOf("</think>");
                    if (think >= 0) {
                        content = content.substring(think + "</think>".length());
                    }
                    prompt.append("<｜Assistant｜>").append(content)
                            .append("<｜end▁of▁sentence｜>");
                }
            }
            return prompt.append("<｜Assistant｜>").toString();
        });
    }

    /**
     * Qwen3 in non-thinking mode: ChatML turns, and an empty {@code <think>} block that opens the answer, as Qwen's
     * {@code enable_thinking=False} does.
     */
    static ChatTemplate qwen3NonThinking() {
        return of(messages -> chatMl(new StringBuilder(), messages)
                .append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
                .toString());
    }

    /**
     * SmolLM3 in non-thinking mode ({@code /no_think}). The system messages become the custom instructions of
     * SmolLM3's metadata block; like the original template, that block has no {@code <|im_end|>}.
     */
    static ChatTemplate smolLM3NonThinking() {
        return of(messages -> {
            StringBuilder instructions = new StringBuilder();
            for (ChatMessage message : messages) {
                if (message.role() == Role.SYSTEM) {
                    instructions.append(instructions.isEmpty() ? "" : "\n\n").append(message.content());
                }
            }
            StringBuilder prompt = new StringBuilder("<|im_start|>system\n## Metadata\n\n")
                    .append("Knowledge Cutoff Date: June 2025\n")
                    .append("Today Date: ")
                    .append(LocalDate.now().format(DateTimeFormatter.ofPattern("d MMMM yyyy", Locale.ENGLISH)))
                    .append("\n")
                    .append("Reasoning Mode: /no_think\n\n")
                    .append("## Custom Instructions\n\n")
                    .append(instructions.isEmpty()
                            ? "You are a helpful AI assistant named SmolLM, trained by Hugging Face."
                            : instructions)
                    .append("\n\n");
            List<ChatMessage> turns = messages.stream().filter(message -> message.role() != Role.SYSTEM).toList();
            return chatMl(prompt, turns).append("<|im_start|>assistant\n<think>\n\n</think>\n").toString();
        });
    }

    private static StringBuilder chatMl(StringBuilder prompt, List<ChatMessage> messages) {
        for (ChatMessage message : messages) {
            prompt.append("<|im_start|>").append(role(message)).append("\n")
                    .append(message.content()).append("<|im_end|>\n");
        }
        return prompt;
    }

    private static String role(ChatMessage message) {
        return message.role().name().toLowerCase(Locale.ROOT);
    }

    private static ChatTemplate of(Function<List<ChatMessage>, String> formatter) {
        return new ChatTemplate() {

            @Override
            public String format(String userMessage) {
                return formatter.apply(List.of(ChatMessage.user(userMessage)));
            }

            @Override
            public String format(List<ChatMessage> messages) {
                if (messages == null || messages.isEmpty()) {
                    throw new IllegalArgumentException("messages must not be empty");
                }
                return formatter.apply(messages);
            }
        };
    }
}
