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

public interface Tokenizer {
    EncodedInput encode(String text);
    EncodedInput encode(String text, int maxLength);

    /**
     * Encodes a sentence pair with segment separation.
     *
     * <p>Used by cross-encoder models that take two text inputs (e.g., query + document).
     * The encoding format is {@code [CLS] textA [SEP] textB [SEP]} with
     * {@code tokenTypeIds} set to 0 for textA tokens and 1 for textB tokens.
     *
     * @param textA     the first sentence
     * @param textB     the second sentence
     * @param maxLength maximum total sequence length (including special tokens)
     * @return the encoded input with proper segment ids
     * @throws UnsupportedOperationException if this tokenizer does not support sentence pairs
     */
    default EncodedInput encode(String textA, String textB, int maxLength) {
        throw new UnsupportedOperationException("Sentence pair encoding not supported by this tokenizer");
    }
}
