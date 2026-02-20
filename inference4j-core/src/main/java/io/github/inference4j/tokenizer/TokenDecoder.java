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
 * Converts token IDs back into text.
 *
 * <p>Used by models with a decoder side (autoregressive, encoder-decoder, seq2seq)
 * that produce token IDs as output. The decode operation reverses the tokenizer's
 * encoding: token IDs are mapped back to their string representations and
 * reassembled into readable text.
 *
 * @see Tokenizer
 */
public interface TokenDecoder {

    /**
     * Decodes a sequence of token IDs back into text.
     *
     * @param tokenIds the token IDs to decode
     * @return the decoded text
     */
    String decode(int[] tokenIds);

    /**
     * Decodes a single token ID to its text representation.
     *
     * <p>Useful for streaming output, where tokens are emitted one at a time.
     * Note that individual tokens may not correspond to complete characters
     * (e.g., multi-byte UTF-8 sequences split across tokens).
     *
     * @param tokenId the token ID to decode
     * @return the text fragment for this token
     */
    String decode(int tokenId);
}
