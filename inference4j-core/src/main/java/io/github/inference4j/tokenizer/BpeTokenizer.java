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
 * Tokenizer implementing byte-level Byte-Pair Encoding (BPE) as used by CLIP and GPT-2.
 *
 * <p>BPE was introduced by
 * <a href="https://arxiv.org/abs/1508.07909">Sennrich et al. (2016)</a> and extended
 * to byte-level encoding by GPT-2
 * (<a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">Radford et al., 2019</a>).
 * CLIP uses a variant with lowercased input and {@code </w>} end-of-word markers.
 *
 * <h2>Algorithm</h2>
 * <ol>
 *   <li><b>Pre-tokenization</b> — input text is lowercased, whitespace-normalized, and
 *       split via regex into words, contractions, digits, and punctuation sequences.</li>
 *   <li><b>Byte-level encoding</b> — each byte of the UTF-8 representation is mapped to
 *       a Unicode character via GPT-2's byte-to-unicode table. This ensures every possible
 *       byte sequence maps to valid Unicode strings.</li>
 *   <li><b>BPE merges</b> — starting from character-level tokens (with {@code </w>}
 *       appended to the last character), pairs are iteratively merged according to the
 *       merge priority table loaded from {@code merges.txt}.</li>
 *   <li><b>Vocabulary lookup</b> — merged tokens are mapped to integer IDs via
 *       {@code vocab.json}.</li>
 *   <li><b>Special tokens</b> — BOS ({@code <|startoftext|>}) and EOS
 *       ({@code <|endoftext|>}) are added, and the sequence is padded to
 *       {@code maxLength}.</li>
 * </ol>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * Tokenizer tokenizer = BpeTokenizer.fromFiles(
 *         Path.of("vocab.json"), Path.of("merges.txt"));
 * EncodedInput encoded = tokenizer.encode("a photo of a cat");
 * // encoded.inputIds()      → [49406, 320, 1125, 539, 320, 2368, 49407, 0, ...]
 * // encoded.attentionMask() → [1, 1, 1, 1, 1, 1, 1, 0, ...]
 * }</pre>
 *
 * @see Tokenizer
 * @see EncodedInput
 */
public class BpeTokenizer implements Tokenizer {

    private static final int DEFAULT_MAX_LENGTH = 77;
    private static final String BOS_TOKEN = "<|startoftext|>";
    private static final String EOS_TOKEN = "<|endoftext|>";
    private static final String END_OF_WORD = "</w>";

    private static final Pattern TOKENIZE_PATTERN = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE
    );

    private final Map<String, Integer> vocab;
    private final Map<Pair, Integer> mergeRanks;
    private final int bosId;
    private final int eosId;
    private final Map<Integer, Character> byteToUnicode;

    private BpeTokenizer(Map<String, Integer> vocab, Map<Pair, Integer> mergeRanks) {
        this.vocab = vocab;
        this.mergeRanks = mergeRanks;
        this.bosId = vocab.getOrDefault(BOS_TOKEN, 49406);
        this.eosId = vocab.getOrDefault(EOS_TOKEN, 49407);
        this.byteToUnicode = buildByteToUnicode();
    }

    /**
     * Creates a BPE tokenizer from vocabulary and merge files.
     *
     * @param vocabJson  path to {@code vocab.json} (token string → integer ID)
     * @param mergesTxt  path to {@code merges.txt} (merge rules, one per line)
     * @return a new BPE tokenizer
     * @throws ModelSourceException if the files cannot be read or parsed
     */
    public static BpeTokenizer fromFiles(Path vocabJson, Path mergesTxt) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            Map<String, Integer> vocab = mapper.readValue(
                    vocabJson.toFile(), new TypeReference<>() {});

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

            return new BpeTokenizer(vocab, mergeRanks);
        } catch (IOException e) {
            throw new ModelSourceException(
                    "Failed to load BPE tokenizer files: " + e.getMessage(), e);
        }
    }

    @Override
    public EncodedInput encode(String text) {
        return encode(text, DEFAULT_MAX_LENGTH);
    }

    @Override
    public EncodedInput encode(String text, int maxLength) {
        List<Integer> tokenIds = new ArrayList<>();
        tokenIds.add(bosId);

        text = text.strip().replaceAll("\\s+", " ").toLowerCase();

        Matcher matcher = TOKENIZE_PATTERN.matcher(text);
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

        tokenIds.add(eosId);

        if (tokenIds.size() > maxLength) {
            tokenIds = new ArrayList<>(tokenIds.subList(0, maxLength - 1));
            tokenIds.add(eosId);
        }

        int actualLength = tokenIds.size();
        long[] inputIds = new long[maxLength];
        for (int i = 0; i < actualLength; i++) {
            inputIds[i] = tokenIds.get(i);
        }

        long[] attentionMask = new long[maxLength];
        Arrays.fill(attentionMask, 0, actualLength, 1L);

        long[] tokenTypeIds = new long[maxLength];

        return new EncodedInput(inputIds, attentionMask, tokenTypeIds);
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
        word.add(token.charAt(token.length() - 1) + END_OF_WORD);

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

    private static Map<Integer, Character> buildByteToUnicode() {
        Map<Integer, Character> map = new HashMap<>();

        // Printable ASCII (33-126) and extended Latin (161-172, 174-255) map to themselves
        for (int i = 33; i <= 126; i++) {
            map.put(i, (char) i);
        }
        for (int i = 161; i <= 172; i++) {
            map.put(i, (char) i);
        }
        for (int i = 174; i <= 255; i++) {
            map.put(i, (char) i);
        }

        // Remaining bytes (0-32, 127-160, 173) map to Unicode codepoints 256+
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!map.containsKey(b)) {
                map.put(b, (char) (256 + n));
                n++;
            }
        }

        return map;
    }

    private record Pair(String first, String second) {}
}
