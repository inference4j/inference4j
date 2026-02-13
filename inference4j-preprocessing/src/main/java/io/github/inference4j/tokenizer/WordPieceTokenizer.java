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

import io.github.inference4j.exception.ModelSourceException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Tokenizer implementing the WordPiece algorithm used by BERT and DistilBERT.
 *
 * <p>WordPiece was introduced in the Japanese/Korean segmentation work by
 * <a href="https://arxiv.org/abs/1609.08144">Schuster &amp; Nakajima (2012)</a> and
 * later adopted by BERT
 * (<a href="https://arxiv.org/abs/1810.04805">Devlin et al., 2019</a>) as its
 * subword tokenization strategy. It solves the core vocabulary problem in NLP:
 * a word-level vocabulary either explodes in size or maps rare words to a single
 * {@code [UNK]} token, losing all semantic information. WordPiece finds a middle
 * ground by breaking unknown words into known subword units — for example,
 * {@code "unbelievable"} becomes {@code ["un", "##believ", "##able"]}, preserving
 * morphological structure even for out-of-vocabulary words.
 *
 * <h2>Algorithm</h2>
 * <p>Tokenization proceeds in two stages:
 * <ol>
 *   <li><b>Basic tokenization</b> — the input is lowercased, stripped, and split on
 *       whitespace and Unicode punctuation. Each punctuation character becomes its own
 *       token.</li>
 *   <li><b>WordPiece tokenization</b> — each basic token is segmented into subwords
 *       via greedy longest-match-first against the vocabulary. Non-initial subwords
 *       are prefixed with {@code ##} to distinguish them from whole-word tokens.
 *       If no match is found at all, the entire token maps to {@code [UNK]}.</li>
 * </ol>
 *
 * <p>The final sequence is wrapped with BERT's special tokens:
 * <ul>
 *   <li>Single text: {@code [CLS] tokens [SEP]}</li>
 *   <li>Sentence pair: {@code [CLS] tokensA [SEP] tokensB [SEP]} with
 *       {@code tokenTypeIds} distinguishing the two segments</li>
 * </ul>
 *
 * <h2>Vocabulary format</h2>
 * <p>Expects a plain-text {@code vocab.txt} file with one token per line, where the
 * line number (0-indexed) is the token's integer ID. This is the standard format
 * used by HuggingFace BERT checkpoints.
 *
 * <h2>Limitations</h2>
 * <p>This tokenizer applies unconditional lowercasing, matching the behavior of
 * {@code bert-base-uncased} and {@code distilbert-base-uncased}. It is not suitable
 * for cased models without modification. Models that use BPE (RoBERTa, GPT-2) or
 * SentencePiece (DeBERTa v3) require different tokenizer implementations.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * Tokenizer tokenizer = WordPieceTokenizer.fromVocabFile(Path.of("vocab.txt"));
 * EncodedInput encoded = tokenizer.encode("Hello world!", 128);
 * // encoded.inputIds()    → [101, 7592, 2088, 999, 102]
 * // encoded.attentionMask() → [1, 1, 1, 1, 1]
 * // encoded.tokenTypeIds()  → [0, 0, 0, 0, 0]
 * }</pre>
 *
 * @see Tokenizer
 * @see EncodedInput
 */
public class WordPieceTokenizer implements Tokenizer {

    private static final String CLS_TOKEN = "[CLS]";
    private static final String SEP_TOKEN = "[SEP]";
    private static final String UNK_TOKEN = "[UNK]";
    private static final String SUBWORD_PREFIX = "##";
    private static final int DEFAULT_MAX_LENGTH = 512;

    private final Map<String, Integer> vocab;
    private final int clsId;
    private final int sepId;
    private final int unkId;

    private WordPieceTokenizer(Map<String, Integer> vocab) {
        this.vocab = vocab;
        this.clsId = vocab.getOrDefault(CLS_TOKEN, 0);
        this.sepId = vocab.getOrDefault(SEP_TOKEN, 0);
        this.unkId = vocab.getOrDefault(UNK_TOKEN, 0);
    }

    public static WordPieceTokenizer fromVocabFile(Path vocabPath) {
        try {
            List<String> lines = Files.readAllLines(vocabPath);
            Map<String, Integer> vocab = new LinkedHashMap<>();
            for (int i = 0; i < lines.size(); i++) {
                String token = lines.get(i).trim();
                if (!token.isEmpty()) {
                    vocab.put(token, i);
                }
            }
            return new WordPieceTokenizer(vocab);
        } catch (IOException e) {
            throw new ModelSourceException(
                    "Failed to load vocabulary from " + vocabPath + ": " + e.getMessage(), e);
        }
    }

    @Override
    public EncodedInput encode(String text) {
        return encode(text, DEFAULT_MAX_LENGTH);
    }

    @Override
    public EncodedInput encode(String text, int maxLength) {
        List<String> basicTokens = basicTokenize(text);

        List<Integer> tokenIds = new ArrayList<>();
        tokenIds.add(clsId);

        for (String token : basicTokens) {
            tokenIds.addAll(wordPieceTokenize(token));
        }

        tokenIds.add(sepId);

        if (tokenIds.size() > maxLength) {
            tokenIds = new ArrayList<>(tokenIds.subList(0, maxLength - 1));
            tokenIds.add(sepId);
        }

        int length = tokenIds.size();
        long[] inputIds = tokenIds.stream().mapToLong(Integer::longValue).toArray();
        long[] attentionMask = new long[length];
        Arrays.fill(attentionMask, 1);
        long[] tokenTypeIds = new long[length];

        return new EncodedInput(inputIds, attentionMask, tokenTypeIds);
    }

    @Override
    public EncodedInput encode(String textA, String textB, int maxLength) {
        List<Integer> idsA = tokenizeToIds(textA);
        List<Integer> idsB = tokenizeToIds(textB);

        // [CLS] textA [SEP] textB [SEP] = idsA.size + idsB.size + 3
        int specialTokens = 3;
        int available = maxLength - specialTokens;
        if (available < 0) {
            available = 0;
        }

        // Truncate the longer sequence first (textB first, as is convention)
        int lenA = idsA.size();
        int lenB = idsB.size();
        while (lenA + lenB > available) {
            if (lenB > lenA) {
                lenB--;
            } else {
                lenA--;
            }
        }

        List<Integer> tokenIds = new ArrayList<>(lenA + lenB + specialTokens);
        tokenIds.add(clsId);
        tokenIds.addAll(idsA.subList(0, lenA));
        tokenIds.add(sepId);
        tokenIds.addAll(idsB.subList(0, lenB));
        tokenIds.add(sepId);

        int length = tokenIds.size();
        long[] inputIds = tokenIds.stream().mapToLong(Integer::longValue).toArray();
        long[] attentionMask = new long[length];
        Arrays.fill(attentionMask, 1);

        // Segment A: [CLS] + idsA + [SEP] = 1 + lenA + 1
        // Segment B: idsB + [SEP] = lenB + 1
        long[] tokenTypeIds = new long[length];
        int segmentBStart = 1 + lenA + 1; // after [CLS] + textA tokens + [SEP]
        for (int i = segmentBStart; i < length; i++) {
            tokenTypeIds[i] = 1;
        }

        return new EncodedInput(inputIds, attentionMask, tokenTypeIds);
    }

    private List<Integer> tokenizeToIds(String text) {
        List<String> basicTokens = basicTokenize(text);
        List<Integer> ids = new ArrayList<>();
        for (String token : basicTokens) {
            ids.addAll(wordPieceTokenize(token));
        }
        return ids;
    }

    private List<String> basicTokenize(String text) {
        text = text.toLowerCase().strip();
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        for (char c : text.toCharArray()) {
            if (Character.isWhitespace(c)) {
                if (!current.isEmpty()) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
            } else if (isPunctuation(c)) {
                if (!current.isEmpty()) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
                tokens.add(String.valueOf(c));
            } else {
                current.append(c);
            }
        }

        if (!current.isEmpty()) {
            tokens.add(current.toString());
        }

        return tokens;
    }

    private List<Integer> wordPieceTokenize(String token) {
        List<Integer> ids = new ArrayList<>();
        int start = 0;

        while (start < token.length()) {
            int end = token.length();
            boolean found = false;

            while (start < end) {
                String substr = (start > 0 ? SUBWORD_PREFIX : "") + token.substring(start, end);
                if (vocab.containsKey(substr)) {
                    ids.add(vocab.get(substr));
                    found = true;
                    break;
                }
                end--;
            }

            if (!found) {
                ids.add(unkId);
                break;
            }

            start = end;
        }

        return ids;
    }

    private boolean isPunctuation(char c) {
        int type = Character.getType(c);
        return type == Character.CONNECTOR_PUNCTUATION
                || type == Character.DASH_PUNCTUATION
                || type == Character.END_PUNCTUATION
                || type == Character.FINAL_QUOTE_PUNCTUATION
                || type == Character.INITIAL_QUOTE_PUNCTUATION
                || type == Character.OTHER_PUNCTUATION
                || type == Character.START_PUNCTUATION;
    }
}
