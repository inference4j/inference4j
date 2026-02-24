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

import java.nio.file.Path;
import java.util.List;

/**
 * Strategy for constructing a tokenizer and decoder from a model directory.
 *
 * <p>Each tokenizer family (GPT-2 BPE, SentencePiece BPE, WordPiece) declares
 * which files it needs and how to build itself from a resolved model directory.
 * This allows the text generator builder to delegate tokenizer construction
 * without knowing the specific tokenizer type.
 *
 * @see DecodingBpeTokenizer#provider()
 * @see SentencePieceBpeTokenizer#provider()
 */
public interface TokenizerProvider {

    /**
     * Files this tokenizer needs in the model directory
     * (e.g., {@code "vocab.json"}, {@code "merges.txt"}).
     */
    List<String> requiredFiles();

    /**
     * Builds a tokenizer and decoder from a resolved model directory.
     *
     * @param modelDir    the directory containing model files
     * @param addedTokens special tokens to register (e.g., {@code <|im_start|>})
     * @return the constructed tokenizer and decoder pair
     */
    TokenizerAndDecoder create(Path modelDir, List<String> addedTokens);

    /**
     * A paired tokenizer and decoder, typically the same object.
     */
    record TokenizerAndDecoder(Tokenizer tokenizer, TokenDecoder decoder) {
    }
}
