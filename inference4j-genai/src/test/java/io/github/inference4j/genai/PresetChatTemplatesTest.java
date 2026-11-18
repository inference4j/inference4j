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
import io.github.inference4j.generation.ChatTemplate;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class PresetChatTemplatesTest {

    private static final List<ChatMessage> CONVERSATION = List.of(
            ChatMessage.system("Be brief."),
            ChatMessage.user("Hi"),
            ChatMessage.assistant("Hello."),
            ChatMessage.user("Bye"));

    @Test
    void singleMessagesAreFormattedAsBefore() {
        assertThat(ModelSources.phi3Mini().chatTemplate().format("Hi"))
                .isEqualTo("<|user|>\nHi<|end|>\n<|assistant|>\n");
        assertThat(ModelSources.phi4Mini().chatTemplate().format("Hi"))
                .isEqualTo("<|user|>Hi<|end|><|assistant|>");
        assertThat(ModelSources.deepSeekR1_1_5B().chatTemplate().format("Hi"))
                .isEqualTo("<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜>");
        assertThat(ModelSources.qwen3_0_6B().chatTemplate().format("Hi"))
                .isEqualTo("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        assertThat(ModelSources.smolLM3_3B().chatTemplate().format("Hi"))
                .startsWith("<|im_start|>system\n## Metadata\n\n")
                .contains("Reasoning Mode: /no_think\n\n## Custom Instructions\n\n"
                        + "You are a helpful AI assistant named SmolLM, trained by Hugging Face.\n\n"
                        + "<|im_start|>user\nHi<|im_end|>\n")
                .endsWith("<|im_start|>assistant\n<think>\n\n</think>\n");
    }

    @Test
    void phiFormatsEveryRole() {
        assertThat(ModelSources.phi3Mini().chatTemplate().format(CONVERSATION)).isEqualTo(
                "<|system|>\nBe brief.<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\nHello.<|end|>\n"
                        + "<|user|>\nBye<|end|>\n<|assistant|>\n");
        assertThat(ModelSources.phi4Mini().chatTemplate().format(CONVERSATION)).isEqualTo(
                "<|system|>Be brief.<|end|><|user|>Hi<|end|><|assistant|>Hello.<|end|><|user|>Bye<|end|><|assistant|>");
    }

    @Test
    void deepSeekPutsTheSystemPromptFirstAndDropsEarlierReasoning() {
        List<ChatMessage> conversation = List.of(
                ChatMessage.system("Be brief."),
                ChatMessage.user("Hi"),
                ChatMessage.assistant("<think>greet</think>Hello."),
                ChatMessage.user("Bye"));

        assertThat(ModelSources.deepSeekR1_1_5B().chatTemplate().format(conversation)).isEqualTo(
                "<｜begin▁of▁sentence｜>Be brief.<｜User｜>Hi"
                        + "<｜Assistant｜>Hello.<｜end▁of▁sentence｜>"
                        + "<｜User｜>Bye<｜Assistant｜>");
    }

    @Test
    void qwenUsesChatMlAndOpensAnEmptyThinkBlock() {
        assertThat(ModelSources.qwen3_0_6B().chatTemplate().format(CONVERSATION)).isEqualTo(
                "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
                        + "<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n"
                        + "<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    @Test
    void smolLmPutsSystemMessagesInTheCustomInstructions() {
        assertThat(ModelSources.smolLM3_3B().chatTemplate().format(CONVERSATION))
                .contains("## Custom Instructions\n\nBe brief.\n\n<|im_start|>user\nHi<|im_end|>\n"
                        + "<|im_start|>assistant\nHello.<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n")
                .doesNotContain("<|im_start|>system\nBe brief.")
                .doesNotContain("named SmolLM");
    }

    @Test
    void emptyConversationIsRejected() {
        ChatTemplate template = ModelSources.phi4Mini().chatTemplate();

        assertThatThrownBy(() -> template.format(List.of())).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void lambdaTemplatesFormatSingleMessagesOnly() {
        ChatTemplate template = message -> "<|user|>" + message;

        assertThatThrownBy(() -> template.format(CONVERSATION)).isInstanceOf(UnsupportedOperationException.class);
    }
}
