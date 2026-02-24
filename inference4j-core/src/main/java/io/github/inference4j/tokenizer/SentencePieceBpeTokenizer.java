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

import io.github.inference4j.tokenizer.BpeTokenizer.Pair;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SentencePiece-style BPE tokenizer for models like Gemma and LLaMA.
 *
 * <p>Unlike GPT-2 BPE ({@link BpeTokenizer}), SentencePiece BPE operates directly
 * on Unicode text with a {@code ▁} (U+2581) space prefix convention and {@code <0xNN>}
 * byte fallback tokens for characters not in the vocabulary. There is no pre-tokenization
 * regex — SentencePiece handles word boundaries via the space prefix.
 *
 * <h2>Encoding</h2>
 * <ol>
 *   <li>Prepend {@code ▁} and replace all spaces with {@code ▁}</li>
 *   <li>Split on added tokens (special tokens preserved atomically)</li>
 *   <li>For non-special segments: split into characters, apply BPE merges</li>
 *   <li>Unmapped characters → UTF-8 bytes → {@code <0xNN>} token IDs</li>
 * </ol>
 *
 * <h2>Decoding</h2>
 * <ol>
 *   <li>Reverse vocab lookup (ID → string)</li>
 *   <li>Skip special token IDs</li>
 *   <li>Detect {@code <0xNN>} tokens → accumulate raw bytes → decode as UTF-8</li>
 *   <li>Replace {@code ▁} with space, strip leading space</li>
 * </ol>
 *
 * @see Tokenizer
 * @see TokenDecoder
 */
public class SentencePieceBpeTokenizer implements Tokenizer, TokenDecoder {

    private static final String SPACE_PREFIX = "\u2581";
    private static final java.util.regex.Pattern BYTE_TOKEN_PATTERN =
            java.util.regex.Pattern.compile("<0x([0-9A-Fa-f]{2})>");

    private final Map<String, Integer> vocab;
    private final Map<Pair, Integer> mergeRanks;
    private final Map<Integer, String> reverseVocab;
    private final Set<Integer> specialTokenIds;
    private final Map<String, Integer> addedTokenMap;
    private final Pattern addedTokenPattern;
    private final int defaultMaxLength;

    private final int[] byteFallbackIds;

    private final ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();

    private SentencePieceBpeTokenizer(Builder builder) {
        this.vocab = builder.vocab;
        this.mergeRanks = builder.mergeRanks;
        this.defaultMaxLength = builder.defaultMaxLength;

        this.reverseVocab = new HashMap<>();
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            reverseVocab.put(entry.getValue(), entry.getKey());
        }

        this.byteFallbackIds = new int[256];
        Arrays.fill(byteFallbackIds, -1);
        for (int b = 0; b < 256; b++) {
            String token = String.format("<0x%02X>", b);
            Integer id = vocab.get(token);
            if (id != null) {
                byteFallbackIds[b] = id;
            }
        }

        if (builder.addedTokens.isEmpty()) {
            this.addedTokenMap = Map.of();
            this.addedTokenPattern = null;
            this.specialTokenIds = Set.of();
        } else {
            this.addedTokenMap = new HashMap<>();
            this.specialTokenIds = new HashSet<>();
            StringBuilder patternBuilder = new StringBuilder();
            for (String token : builder.addedTokens) {
                Integer id = vocab.get(token);
                if (id != null) {
                    addedTokenMap.put(token, id);
                    specialTokenIds.add(id);
                    if (patternBuilder.length() > 0) {
                        patternBuilder.append('|');
                    }
                    patternBuilder.append(Pattern.quote(token));
                }
            }
            this.addedTokenPattern = addedTokenMap.isEmpty()
                    ? null
                    : Pattern.compile(patternBuilder.toString());
        }
    }

    /**
     * Creates a provider that builds tokenizers from HuggingFace {@code tokenizer.json} files.
     */
    public static TokenizerProvider provider() {
        return new TokenizerProvider() {
            @Override
            public List<String> requiredFiles() {
                return List.of("tokenizer.json");
            }

            @Override
            public TokenizerAndDecoder create(Path dir, List<String> addedTokens) {
                SentencePieceBpeTokenizer.Builder b =
                        TokenizerJsonParser.parse(dir.resolve("tokenizer.json"));
                for (String t : addedTokens) {
                    b.addedToken(t);
                }
                SentencePieceBpeTokenizer tok = b.build();
                return new TokenizerAndDecoder(tok, tok);
            }
        };
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public EncodedInput encode(String text) {
        return encode(text, defaultMaxLength);
    }

    @Override
    public EncodedInput encode(String text, int maxLength) {
        List<Integer> tokenIds = tokenize(text);

        if (tokenIds.size() > maxLength) {
            tokenIds = new ArrayList<>(tokenIds.subList(0, maxLength));
        }

        int length = tokenIds.size();
        long[] inputIds = new long[length];
        for (int i = 0; i < length; i++) {
            inputIds[i] = tokenIds.get(i);
        }

        long[] attentionMask = new long[length];
        Arrays.fill(attentionMask, 1L);

        long[] tokenTypeIds = new long[length];

        return new EncodedInput(inputIds, attentionMask, tokenTypeIds);
    }

    @Override
    public String decode(int[] tokenIds) {
        StringBuilder sb = new StringBuilder();
        ByteArrayOutputStream pendingBytes = new ByteArrayOutputStream();

        for (int id : tokenIds) {
            if (specialTokenIds.contains(id)) {
                flushBytes(pendingBytes, sb);
                continue;
            }
            String token = reverseVocab.get(id);
            if (token == null) {
                flushBytes(pendingBytes, sb);
                continue;
            }

            Matcher m = BYTE_TOKEN_PATTERN.matcher(token);
            if (m.matches()) {
                pendingBytes.write(Integer.parseInt(m.group(1), 16));
            } else {
                flushBytes(pendingBytes, sb);
                sb.append(token);
            }
        }
        flushBytes(pendingBytes, sb);

        String result = sb.toString().replace(SPACE_PREFIX, " ");
        if (result.startsWith(" ")) {
            result = result.substring(1);
        }
        return result;
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

        Matcher m = BYTE_TOKEN_PATTERN.matcher(token);
        if (m.matches()) {
            byteBuffer.write(Integer.parseInt(m.group(1), 16));
            return tryFlushByteBuffer();
        }

        String prefix = tryForceFlushByteBuffer();
        String text = token.replace(SPACE_PREFIX, " ");
        return prefix + text;
    }

    private String tryFlushByteBuffer() {
        if (byteBuffer.size() == 0) {
            return "";
        }
        byte[] bytes = byteBuffer.toByteArray();
        if (isCompleteUtf8(bytes)) {
            byteBuffer.reset();
            return new String(bytes, StandardCharsets.UTF_8);
        }
        return "";
    }

    private String tryForceFlushByteBuffer() {
        if (byteBuffer.size() == 0) {
            return "";
        }
        byte[] bytes = byteBuffer.toByteArray();
        byteBuffer.reset();
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static boolean isCompleteUtf8(byte[] bytes) {
        int i = 0;
        while (i < bytes.length) {
            int b = bytes[i] & 0xFF;
            int expectedLen;
            if (b < 0x80) {
                expectedLen = 1;
            } else if (b < 0xC0) {
                return false; // unexpected continuation byte
            } else if (b < 0xE0) {
                expectedLen = 2;
            } else if (b < 0xF0) {
                expectedLen = 3;
            } else if (b < 0xF8) {
                expectedLen = 4;
            } else {
                return false;
            }
            if (i + expectedLen > bytes.length) {
                return false; // incomplete sequence
            }
            i += expectedLen;
        }
        return true;
    }

    private List<Integer> tokenize(String text) {
        List<Integer> tokenIds = new ArrayList<>();

        if (addedTokenPattern != null) {
            Matcher addedMatcher = addedTokenPattern.matcher(text);
            int lastEnd = 0;
            boolean isFirst = true;
            while (addedMatcher.find()) {
                if (addedMatcher.start() > lastEnd) {
                    String segment = text.substring(lastEnd, addedMatcher.start());
                    tokenizeBpe(applySpacePrefix(segment, isFirst), tokenIds);
                    isFirst = false;
                }
                tokenIds.add(addedTokenMap.get(addedMatcher.group()));
                lastEnd = addedMatcher.end();
            }
            if (lastEnd < text.length()) {
                String segment = text.substring(lastEnd);
                tokenizeBpe(applySpacePrefix(segment, isFirst), tokenIds);
            }
        } else {
            tokenizeBpe(applySpacePrefix(text, true), tokenIds);
        }

        return tokenIds;
    }

    private static String applySpacePrefix(String text, boolean prependPrefix) {
        String result = text.replace(" ", SPACE_PREFIX);
        if (prependPrefix) {
            result = SPACE_PREFIX + result;
        }
        return result;
    }

    private void tokenizeBpe(String text, List<Integer> tokenIds) {
        if (text.isEmpty()) {
            return;
        }

        List<String> chars = new ArrayList<>();
        for (int i = 0; i < text.length(); ) {
            int cp = text.codePointAt(i);
            chars.add(new String(Character.toChars(cp)));
            i += Character.charCount(cp);
        }

        List<String> bpeTokens = bpe(chars);

        for (String token : bpeTokens) {
            Integer id = vocab.get(token);
            if (id != null) {
                tokenIds.add(id);
            } else {
                byte[] bytes = token.getBytes(StandardCharsets.UTF_8);
                for (byte b : bytes) {
                    int fallbackId = byteFallbackIds[b & 0xFF];
                    if (fallbackId >= 0) {
                        tokenIds.add(fallbackId);
                    }
                }
            }
        }
    }

    private List<String> bpe(List<String> word) {
        if (word.size() <= 1) {
            return word;
        }

        word = new ArrayList<>(word);

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
                int j = indexOf(word, bestPair.first(), i);
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size()));
                    break;
                }
                newWord.addAll(word.subList(i, j));
                i = j;

                if (i < word.size() - 1
                        && word.get(i).equals(bestPair.first())
                        && word.get(i + 1).equals(bestPair.second())) {
                    newWord.add(bestPair.first() + bestPair.second());
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

    private static void flushBytes(ByteArrayOutputStream pending, StringBuilder sb) {
        if (pending.size() > 0) {
            sb.append(pending.toString(StandardCharsets.UTF_8));
            pending.reset();
        }
    }

    public static class Builder {

        private Map<String, Integer> vocab = new LinkedHashMap<>();
        private Map<Pair, Integer> mergeRanks = new LinkedHashMap<>();
        private final List<String> addedTokens = new ArrayList<>();
        private int defaultMaxLength = 8192;

        public Builder vocab(Map<String, Integer> vocab) {
            this.vocab = vocab;
            return this;
        }

        public Builder merges(List<String> mergeLines) {
            this.mergeRanks = new LinkedHashMap<>();
            for (int i = 0; i < mergeLines.size(); i++) {
                String line = mergeLines.get(i).trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(" ", 2);
                if (parts.length == 2) {
                    mergeRanks.put(new Pair(parts[0], parts[1]), i);
                }
            }
            return this;
        }

        public Builder addedToken(String token) {
            this.addedTokens.add(token);
            return this;
        }

        public Builder defaultMaxLength(int maxLength) {
            this.defaultMaxLength = maxLength;
            return this;
        }

        public SentencePieceBpeTokenizer build() {
            return new SentencePieceBpeTokenizer(this);
        }
    }
}
