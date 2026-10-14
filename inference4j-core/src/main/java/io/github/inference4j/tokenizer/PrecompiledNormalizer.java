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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SentencePiece's character normalization, as stored in a HuggingFace {@code tokenizer.json}
 * {@code Precompiled} normalizer.
 *
 * <p>The {@code precompiled_charsmap} is a double-array trie mapping UTF-8 sequences to their
 * normalized form (NFKC plus SentencePiece's own rules, e.g. tabs and zero-width spaces become spaces).
 * Text is normalized one grapheme cluster at a time, falling back to single code points when a
 * cluster has no mapping, exactly as HuggingFace {@code tokenizers} does.
 */
public final class PrecompiledNormalizer implements UnaryOperator<String> {

    private static final Pattern GRAPHEME = Pattern.compile("\\X");

    private final int[] trie;
    private final byte[] normalized;

    private PrecompiledNormalizer(int[] trie, byte[] normalized) {
        this.trie = trie;
        this.normalized = normalized;
    }

    /**
     * Reads a {@code precompiled_charsmap}: a little-endian trie size in bytes, the trie's 32-bit units,
     * then the null-terminated normalized strings the trie points into.
     */
    public static PrecompiledNormalizer fromBase64(String precompiledCharsmap) {
        byte[] bytes = Base64.getDecoder().decode(precompiledCharsmap);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int trieBytes = buffer.getInt();
        if (trieBytes < 0 || trieBytes % 4 != 0 || 4L + trieBytes > bytes.length) {
            throw new IllegalArgumentException("Invalid precompiled_charsmap: trie size " + trieBytes);
        }
        int[] trie = new int[trieBytes / 4];
        for (int i = 0; i < trie.length; i++) {
            trie[i] = buffer.getInt();
        }
        return new PrecompiledNormalizer(trie, Arrays.copyOfRange(bytes, 4 + trieBytes, bytes.length));
    }

    @Override
    public String apply(String text) {
        if (text.isEmpty()) {
            return text;
        }
        StringBuilder out = new StringBuilder(text.length());
        Matcher graphemes = GRAPHEME.matcher(text);
        while (graphemes.find()) {
            String grapheme = graphemes.group();
            byte[] utf8 = grapheme.getBytes(StandardCharsets.UTF_8);
            if (utf8.length < 6) {
                String mapped = transform(utf8);
                if (mapped != null) {
                    out.append(mapped);
                    continue;
                }
            }
            grapheme.codePoints().forEach(codePoint -> {
                String part = new String(Character.toChars(codePoint));
                String mapped = transform(part.getBytes(StandardCharsets.UTF_8));
                out.append(mapped != null ? mapped : part);
            });
        }
        return out.toString();
    }

    /**
     * The normalized form of the shortest prefix of {@code key} found in the trie, or {@code null}. Like
     * {@code tokenizers}, the first match is used even if it covers only part of the key.
     */
    private String transform(byte[] key) {
        int nodePos = offset(trie[0]);
        for (byte b : key) {
            int c = b & 0xFF;
            if (c == 0) {
                break;
            }
            nodePos ^= c;
            if (nodePos < 0 || nodePos >= trie.length) {
                return null;
            }
            int unit = trie[nodePos];
            if (label(unit) != c) {
                return null;
            }
            nodePos ^= offset(unit);
            if (hasLeaf(unit)) {
                return normalizedAt(trie[nodePos] & 0x7FFFFFFF);
            }
        }
        return null;
    }

    private String normalizedAt(int index) {
        int end = index;
        while (end < normalized.length && normalized[end] != 0) {
            end++;
        }
        return new String(normalized, index, end - index, StandardCharsets.UTF_8);
    }

    // Unit layout of a darts-clone double array
    private static boolean hasLeaf(int unit) {
        return ((unit >>> 8) & 1) == 1;
    }

    private static int label(int unit) {
        return unit & 0x800000FF;
    }

    private static int offset(int unit) {
        return (unit >>> 10) << ((unit & (1 << 9)) >>> 6);
    }
}
