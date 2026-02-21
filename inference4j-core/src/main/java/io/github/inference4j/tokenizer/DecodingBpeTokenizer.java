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

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A {@link BpeTokenizer} that also supports decoding token IDs back to text.
 *
 * <p>Extends the base BPE tokenizer with a reverse vocabulary and byte-decoding
 * pipeline. Used by models that produce token IDs as output: autoregressive
 * (GPT-2, Phi), encoder-decoder (Whisper, TrOCR), and seq2seq (T5, BART).
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * DecodingBpeTokenizer tokenizer = DecodingBpeTokenizer.fromFiles(vocabPath, mergesPath);
 *
 * // Encode
 * EncodedInput encoded = tokenizer.encode("Hello world");
 *
 * // Decode
 * String text = tokenizer.decode(new int[]{15496, 995});  // "Hello world"
 *
 * // Single token (for streaming)
 * String fragment = tokenizer.decode(15496);  // "Hello"
 * }</pre>
 *
 * @see BpeTokenizer
 * @see TokenDecoder
 */
public class DecodingBpeTokenizer extends BpeTokenizer implements TokenDecoder {

    private final Map<Integer, String> reverseVocab;
    private final Map<Character, Integer> unicodeToByte;
    private final Set<Integer> specialTokenIds;

    DecodingBpeTokenizer(Builder builder) {
        super(builder);

        this.reverseVocab = new HashMap<>();
        for (Map.Entry<String, Integer> entry : builder.vocab.entrySet()) {
            reverseVocab.put(entry.getValue(), entry.getKey());
        }

        Map<Integer, Character> byteToUnicode = buildByteToUnicode();
        this.unicodeToByte = new HashMap<>();
        for (Map.Entry<Integer, Character> entry : byteToUnicode.entrySet()) {
            unicodeToByte.put(entry.getValue(), entry.getKey());
        }

        this.specialTokenIds = new HashSet<>();
        if (builder.bosToken != null) {
            Integer id = builder.vocab.get(builder.bosToken);
            if (id != null) {
                specialTokenIds.add(id);
            }
        }
        if (builder.eosToken != null) {
            Integer id = builder.vocab.get(builder.eosToken);
            if (id != null) {
                specialTokenIds.add(id);
            }
        }
        for (String addedToken : builder.addedTokens) {
            Integer id = builder.vocab.get(addedToken);
            if (id != null) {
                specialTokenIds.add(id);
            }
        }
    }

    /**
     * Creates a decoding BPE tokenizer with GPT-2 defaults.
     *
     * @param vocabJson path to {@code vocab.json}
     * @param mergesTxt path to {@code merges.txt}
     * @return a new decoding BPE tokenizer
     */
    public static DecodingBpeTokenizer fromFiles(Path vocabJson, Path mergesTxt) {
        return new DecodingBpeTokenizer(
                new Builder(loadVocab(vocabJson), loadMerges(mergesTxt)));
    }

    /**
     * Creates a decoding BPE tokenizer from a pre-configured builder.
     *
     * <pre>{@code
     * DecodingBpeTokenizer tokenizer = DecodingBpeTokenizer.from(
     *         BpeTokenizer.builder(vocabPath, mergesPath)
     *                 .eosToken("<|endoftext|>"));
     * }</pre>
     *
     * @param builder a configured BPE tokenizer builder
     * @return a new decoding BPE tokenizer
     */
    public static DecodingBpeTokenizer from(Builder builder) {
        return new DecodingBpeTokenizer(builder);
    }

    @Override
    public String decode(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();
        for (int id : tokenIds) {
            if (specialTokenIds.contains(id)) {
                continue;
            }
            String token = reverseVocab.get(id);
            if (token == null) {
                continue;
            }
            sb.append(token);
        }
        return reverseBytePairEncoding(sb.toString());
    }

    @Override
    public String decode(int tokenId) {
        if (specialTokenIds.contains(tokenId)) {
            return "";
        }
        String token = reverseVocab.get(tokenId);
        if (token == null) {
            return "";
        }
        return reverseBytePairEncoding(token);
    }

    private String reverseBytePairEncoding(String bpeString) {
        byte[] bytes = new byte[bpeString.length()];
        int byteCount = 0;
        for (int i = 0; i < bpeString.length(); i++) {
            Integer byteVal = unicodeToByte.get(bpeString.charAt(i));
            if (byteVal != null) {
                bytes[byteCount++] = (byte) (int) byteVal;
            }
        }
        return new String(bytes, 0, byteCount, StandardCharsets.UTF_8);
    }
}
