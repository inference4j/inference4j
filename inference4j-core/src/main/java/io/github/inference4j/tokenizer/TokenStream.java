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

package io.github.inference4j.tokenizer;

/**
 * Decodes the tokens of one generation as they arrive, one at a time.
 *
 * <p>Some characters span several tokens: SentencePiece byte-fallback tokens such as {@code <0xC3> <0xA9>}
 * for {@code é}, or a byte-level BPE token that ends halfway through an emoji. The stream holds those
 * bytes until the character is complete, so {@link #next(int)} can return an empty string.
 *
 * <p>Create one stream per generation with {@link TokenDecoder#newStream()} and use it from one thread.
 * The tokenizer that created it stays thread-safe and can be shared.
 */
public interface TokenStream {

    /**
     * Adds one token and returns the text that is now complete.
     *
     * @param tokenId the next generated token
     * @return the newly completed text; empty while the token's bytes are held. A special token adds no
     *         text of its own but ends any character still held, as {@code U+FFFD} if it is unfinished
     */
    String next(int tokenId);

    /**
     * Returns any bytes still held, as text. Call it once at the end of a generation; an incomplete
     * character becomes U+FFFD.
     *
     * @return the remaining text, usually empty
     */
    String flush();
}
