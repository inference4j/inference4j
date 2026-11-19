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

package io.github.inference4j.generation;

import java.util.Objects;

/**
 * One message of a conversation, for {@link ChatTemplate#format(java.util.List)}.
 *
 * @param role    who wrote the message
 * @param content the text of the message
 */
public record ChatMessage(Role role, String content) {

    public ChatMessage {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(content, "content");
    }

    public static ChatMessage system(String content) {
        return new ChatMessage(Role.SYSTEM, content);
    }

    public static ChatMessage user(String content) {
        return new ChatMessage(Role.USER, content);
    }

    public static ChatMessage assistant(String content) {
        return new ChatMessage(Role.ASSISTANT, content);
    }

    /** Who wrote a message. */
    public enum Role {
        /** Instructions for the model, usually first. */
        SYSTEM,
        /** A message from the user. */
        USER,
        /** An earlier reply of the model. */
        ASSISTANT
    }
}
