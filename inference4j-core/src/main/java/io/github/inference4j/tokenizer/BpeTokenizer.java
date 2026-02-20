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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.inference4j.exception.ModelSourceException;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Configurable byte-level Byte-Pair Encoding (BPE) tokenizer.
 *
 * <p>BPE was introduced by
 * <a href="https://arxiv.org/abs/1508.07909">Sennrich et al. (2016)</a> and extended
 * to byte-level encoding by GPT-2
 * (<a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">Radford et al., 2019</a>).
 * This implementation supports the full family of models that use
 * {@code vocab.json} + {@code merges.txt}: GPT-2, CLIP, Whisper, RoBERTa,
 * Phi-1/2, GPT-J, StarCoder, and others.
 *
 * <h2>Algorithm</h2>
 * <ol>
 *   <li><b>Pre-tokenization</b> — input text is optionally lowercased, whitespace-normalized,
 *       and split via a configurable regex into words, contractions, digits, and
 *       punctuation sequences.</li>
 *   <li><b>Byte-level encoding</b> — each byte of the UTF-8 representation is mapped to
 *       a Unicode character via GPT-2's byte-to-unicode table.</li>
 *   <li><b>BPE merges</b> — starting from character-level tokens (with an optional
 *       end-of-word marker appended to the last character), pairs are iteratively merged
 *       according to the merge priority table loaded from {@code merges.txt}.</li>
 *   <li><b>Vocabulary lookup</b> — merged tokens are mapped to integer IDs via
 *       {@code vocab.json}.</li>
 *   <li><b>Special tokens</b> — BOS/EOS tokens are optionally added, and the sequence
 *       is optionally padded to {@code maxLength}.</li>
 * </ol>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // GPT-2 style (default — no lowercasing, no padding, no BOS/EOS)
 * Tokenizer tokenizer = BpeTokenizer.fromFiles(vocabPath, mergesPath);
 *
 * // Custom configuration
 * Tokenizer tokenizer = BpeTokenizer.builder(vocabPath, mergesPath)
 *         .lowercase(true)
 *         .endOfWordMarker("</w>")
 *         .bosToken("<|startoftext|>")
 *         .eosToken("<|endoftext|>")
 *         .pad(true)
 *         .defaultMaxLength(77)
 *         .build();
 * }</pre>
 *
 * @see Tokenizer
 * @see EncodedInput
 */
public class BpeTokenizer implements Tokenizer {

    /**
     * GPT-2 pre-tokenization pattern. Splits on contractions, words (with optional
     * leading space), numbers, punctuation, and whitespace.
     */
    public static final Pattern GPT2_PATTERN = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );


    private final Map<String, Integer> vocab;
    private final Map<Pair, Integer> mergeRanks;
    private final Map<Integer, Character> byteToUnicode;
    private final Pattern pattern;
    private final boolean lowercase;
    private final String endOfWordMarker;
    private final Integer bosId;
    private final Integer eosId;
    private final boolean pad;
    private final int defaultMaxLength;

    BpeTokenizer(Builder builder) {
        this.vocab = builder.vocab;
        this.mergeRanks = builder.mergeRanks;
        this.byteToUnicode = buildByteToUnicode();
        this.pattern = builder.pattern;
        this.lowercase = builder.lowercase;
        this.endOfWordMarker = builder.endOfWordMarker;
        this.bosId = builder.bosToken != null ? vocab.get(builder.bosToken) : null;
        this.eosId = builder.eosToken != null ? vocab.get(builder.eosToken) : null;
        this.pad = builder.pad;
        this.defaultMaxLength = builder.defaultMaxLength;
    }

    /**
     * Creates a BPE tokenizer with GPT-2 defaults (no lowercasing, no end-of-word
     * marker, no padding, no BOS/EOS wrapping).
     *
     * @param vocabJson  path to {@code vocab.json} (token string to integer ID)
     * @param mergesTxt  path to {@code merges.txt} (merge rules, one per line)
     * @return a new BPE tokenizer with GPT-2 defaults
     * @throws ModelSourceException if the files cannot be read or parsed
     */
    public static BpeTokenizer fromFiles(Path vocabJson, Path mergesTxt) {
        return builder(vocabJson, mergesTxt).build();
    }

    /**
     * Creates a builder for a BPE tokenizer.
     *
     * @param vocabJson  path to {@code vocab.json}
     * @param mergesTxt  path to {@code merges.txt}
     * @return a new builder
     * @throws ModelSourceException if the files cannot be read or parsed
     */
    public static Builder builder(Path vocabJson, Path mergesTxt) {
        return new Builder(loadVocab(vocabJson), loadMerges(mergesTxt));
    }

    @Override
    public EncodedInput encode(String text) {
        return encode(text, defaultMaxLength);
    }

    @Override
    public EncodedInput encode(String text, int maxLength) {
        List<Integer> tokenIds = tokenize(text);

        if (bosId != null) {
            tokenIds.add(0, bosId);
        }
        if (eosId != null) {
            tokenIds.add(eosId);
        }

        if (tokenIds.size() > maxLength) {
            tokenIds = new ArrayList<>(tokenIds.subList(0, maxLength - (eosId != null ? 1 : 0)));
            if (eosId != null) {
                tokenIds.add(eosId);
            }
        }

        int actualLength = tokenIds.size();
        int arrayLength = pad ? maxLength : actualLength;

        long[] inputIds = new long[arrayLength];
        for (int i = 0; i < actualLength; i++) {
            inputIds[i] = tokenIds.get(i);
        }

        long[] attentionMask = new long[arrayLength];
        Arrays.fill(attentionMask, 0, actualLength, 1L);

        long[] tokenTypeIds = new long[arrayLength];

        return new EncodedInput(inputIds, attentionMask, tokenTypeIds);
    }

    /**
     * Tokenizes text into token IDs without adding special tokens, padding, or truncation.
     *
     * @param text the input text
     * @return mutable list of token IDs
     */
    List<Integer> tokenize(String text) {
        List<Integer> tokenIds = new ArrayList<>();

        String processed = text.strip().replaceAll("\\s+", " ");
        if (lowercase) {
            processed = processed.toLowerCase();
        }

        Matcher matcher = pattern.matcher(processed);
        while (matcher.find()) {
            String token = matcher.group();
            String byteEncoded = byteEncode(token);
            List<String> bpeTokens = bpe(byteEncoded);
            for (String bpeToken : bpeTokens) {
                Integer id = vocab.get(bpeToken);
                if (id != null) {
                    tokenIds.add(id);
                }
            }
        }

        return tokenIds;
    }

    private String byteEncode(String token) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = token.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.append(byteToUnicode.get(b & 0xFF));
        }
        return sb.toString();
    }

    private List<String> bpe(String token) {
        List<String> word = new ArrayList<>();
        for (int i = 0; i < token.length() - 1; i++) {
            word.add(String.valueOf(token.charAt(i)));
        }
        if (endOfWordMarker != null) {
            word.add(token.charAt(token.length() - 1) + endOfWordMarker);
        } else {
            word.add(String.valueOf(token.charAt(token.length() - 1)));
        }

        if (word.size() == 1) {
            return word;
        }

        while (true) {
            Pair bestPair = null;
            int bestRank = Integer.MAX_VALUE;

            for (int i = 0; i < word.size() - 1; i++) {
                Pair pair = new Pair(word.get(i), word.get(i + 1));
                Integer rank = mergeRanks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestPair = pair;
                }
            }

            if (bestPair == null) {
                break;
            }

            List<String> newWord = new ArrayList<>();
            int i = 0;
            while (i < word.size()) {
                int j = indexOf(word, bestPair.first, i);
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size()));
                    break;
                }
                newWord.addAll(word.subList(i, j));
                i = j;

                if (i < word.size() - 1
                        && word.get(i).equals(bestPair.first)
                        && word.get(i + 1).equals(bestPair.second)) {
                    newWord.add(bestPair.first + bestPair.second);
                    i += 2;
                } else {
                    newWord.add(word.get(i));
                    i += 1;
                }
            }

            word = newWord;
            if (word.size() == 1) {
                break;
            }
        }

        return word;
    }

    private static int indexOf(List<String> list, String target, int fromIndex) {
        for (int i = fromIndex; i < list.size(); i++) {
            if (list.get(i).equals(target)) {
                return i;
            }
        }
        return -1;
    }

    static Map<Integer, Character> buildByteToUnicode() {
        Map<Integer, Character> map = new HashMap<>();

        for (int i = 33; i <= 126; i++) {
            map.put(i, (char) i);
        }
        for (int i = 161; i <= 172; i++) {
            map.put(i, (char) i);
        }
        for (int i = 174; i <= 255; i++) {
            map.put(i, (char) i);
        }

        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!map.containsKey(b)) {
                map.put(b, (char) (256 + n));
                n++;
            }
        }

        return map;
    }

    static Map<String, Integer> loadVocab(Path vocabJson) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            return mapper.readValue(vocabJson.toFile(), new TypeReference<>() {});
        } catch (IOException e) {
            throw new ModelSourceException(
                    "Failed to load BPE vocabulary: " + e.getMessage(), e);
        }
    }

    static Map<Pair, Integer> loadMerges(Path mergesTxt) {
        try {
            List<String> mergeLines = Files.readAllLines(mergesTxt);
            Map<Pair, Integer> mergeRanks = new LinkedHashMap<>();
            for (int i = 1; i < mergeLines.size(); i++) {
                String line = mergeLines.get(i).trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(" ", 2);
                if (parts.length == 2) {
                    mergeRanks.put(new Pair(parts[0], parts[1]), i - 1);
                }
            }
            return mergeRanks;
        } catch (IOException e) {
            throw new ModelSourceException(
                    "Failed to load BPE merges: " + e.getMessage(), e);
        }
    }

    record Pair(String first, String second) {}

    public static class Builder {

        final Map<String, Integer> vocab;
        final Map<Pair, Integer> mergeRanks;
        Pattern pattern = GPT2_PATTERN;
        boolean lowercase = false;
        String endOfWordMarker = null;
        String bosToken = null;
        String eosToken = null;
        boolean pad = false;
        int defaultMaxLength = 512;

        Builder(Map<String, Integer> vocab, Map<Pair, Integer> mergeRanks) {
            this.vocab = vocab;
            this.mergeRanks = mergeRanks;
        }

        public Builder pattern(Pattern pattern) {
            this.pattern = pattern;
            return this;
        }

        public Builder lowercase(boolean lowercase) {
            this.lowercase = lowercase;
            return this;
        }

        public Builder endOfWordMarker(String endOfWordMarker) {
            this.endOfWordMarker = endOfWordMarker;
            return this;
        }

        public Builder bosToken(String bosToken) {
            this.bosToken = bosToken;
            return this;
        }

        public Builder eosToken(String eosToken) {
            this.eosToken = eosToken;
            return this;
        }

        public Builder pad(boolean pad) {
            this.pad = pad;
            return this;
        }

        public Builder defaultMaxLength(int defaultMaxLength) {
            this.defaultMaxLength = defaultMaxLength;
            return this;
        }

        public BpeTokenizer build() {
            return new BpeTokenizer(this);
        }
    }
}
