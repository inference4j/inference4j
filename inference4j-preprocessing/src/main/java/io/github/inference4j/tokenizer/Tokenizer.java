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
 * Converts raw text into the numerical representation expected by transformer models.
 *
 * <p>Transformer architectures do not operate on strings directly — they require
 * sequences of integer token IDs, each mapped from a fixed vocabulary learned during
 * pre-training. A tokenizer bridges this gap: it splits text into tokens, maps each
 * token to its vocabulary index, and wraps the result with the special tokens and
 * metadata tensors ({@code attention_mask}, {@code token_type_ids}) that the model
 * expects as input.
 *
 * <p>Different model families use different tokenization algorithms:
 * <ul>
 *   <li><b>WordPiece</b> (BERT, DistilBERT) — greedy longest-match subword splitting
 *       with a {@code ##} continuation prefix. See {@link WordPieceTokenizer}.</li>
 *   <li><b>Byte-Pair Encoding (BPE)</b> (RoBERTa, GPT-2) — iteratively merges the
 *       most frequent character pairs into subwords.</li>
 *   <li><b>SentencePiece</b> (DeBERTa v3, T5) — a language-independent subword
 *       segmenter that operates on raw Unicode without pre-tokenization.</li>
 * </ul>
 *
 * <p>This interface abstracts over the algorithm so that model wrappers can accept
 * any compatible tokenizer. A model wrapper calls {@link #encode(String, int)} and
 * feeds the resulting {@link EncodedInput} tensors directly into an
 * {@link io.github.inference4j.InferenceSession}.
 *
 * @see EncodedInput
 * @see WordPieceTokenizer
 */
public interface Tokenizer {
    /**
     * Encodes text using the tokenizer's default maximum sequence length.
     *
     * @param text the input text
     * @return the encoded input tensors
     */
    EncodedInput encode(String text);

    /**
     * Encodes text, truncating to the specified maximum sequence length.
     *
     * @param text      the input text
     * @param maxLength maximum total sequence length (including special tokens)
     * @return the encoded input tensors
     */
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
