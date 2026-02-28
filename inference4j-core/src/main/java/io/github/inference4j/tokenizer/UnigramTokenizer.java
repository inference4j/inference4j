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

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SentencePiece-style Unigram tokenizer with Viterbi decoding.
 *
 * <p>Unlike BPE ({@link SentencePieceBpeTokenizer}), the Unigram algorithm assigns a
 * log-probability score to every token in the vocabulary and uses dynamic programming
 * (Viterbi) to find the segmentation that maximizes the total score. This produces
 * optimal segmentations for models like Flan-T5, CoEdit, and other T5-family models
 * that use SentencePiece Unigram tokenization.
 *
 * <h2>Encoding</h2>
 * <ol>
 *   <li>Prepend {@code ▁} and replace all spaces with {@code ▁}</li>
 *   <li>Split on added tokens (special tokens preserved atomically)</li>
 *   <li>For non-special segments: run Viterbi to find optimal segmentation</li>
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
public class UnigramTokenizer implements Tokenizer, TokenDecoder {

	private static final String SPACE_PREFIX = "\u2581";
	private static final Pattern BYTE_TOKEN_PATTERN =
			Pattern.compile("<0x([0-9A-Fa-f]{2})>");

	private final Map<String, Integer> vocab;
	private final float[] scores;
	private final Map<Integer, String> reverseVocab;
	private final Set<Integer> specialTokenIds;
	private final Map<String, Integer> addedTokenMap;
	private final Pattern addedTokenPattern;
	private final int defaultMaxLength;
	private final int maxTokenLength;

	private final int[] byteFallbackIds;

	private final ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();

	private UnigramTokenizer(Builder builder) {
		this.vocab = builder.vocab;
		this.scores = builder.scores;
		this.defaultMaxLength = builder.defaultMaxLength;

		int maxLen = 0;
		for (String token : vocab.keySet()) {
			maxLen = Math.max(maxLen, token.length());
		}
		this.maxTokenLength = maxLen;

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
	 * Creates a provider that builds Unigram tokenizers from HuggingFace
	 * {@code tokenizer.json} files.
	 */
	public static TokenizerProvider provider() {
		return new TokenizerProvider() {
			@Override
			public List<String> requiredFiles() {
				return List.of("tokenizer.json");
			}

			@Override
			public TokenizerAndDecoder create(Path dir, List<String> addedTokens) {
				UnigramTokenizer.Builder b =
						TokenizerJsonParser.parseUnigram(dir.resolve("tokenizer.json"));
				for (String t : addedTokens) {
					b.addedToken(t);
				}
				UnigramTokenizer tok = b.build();
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
				return false;
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
				return false;
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
					tokenizeViterbi(applySpacePrefix(segment, isFirst), tokenIds);
					isFirst = false;
				}
				tokenIds.add(addedTokenMap.get(addedMatcher.group()));
				lastEnd = addedMatcher.end();
			}
			if (lastEnd < text.length()) {
				String segment = text.substring(lastEnd);
				tokenizeViterbi(applySpacePrefix(segment, isFirst), tokenIds);
			}
		} else {
			tokenizeViterbi(applySpacePrefix(text, true), tokenIds);
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

	/**
	 * Segments text using the Viterbi algorithm to find the highest-scoring
	 * tokenization according to the Unigram model's log-probability scores.
	 */
	private void tokenizeViterbi(String text, List<Integer> tokenIds) {
		int len = text.length();
		if (len == 0) {
			return;
		}

		// Forward pass: dynamic programming
		float[] bestScore = new float[len + 1];
		int[] bestPrev = new int[len + 1];
		int[] bestTokenId = new int[len + 1];

		Arrays.fill(bestScore, Float.NEGATIVE_INFINITY);
		bestScore[0] = 0.0f;
		Arrays.fill(bestPrev, -1);
		Arrays.fill(bestTokenId, -1);

		for (int i = 1; i <= len; i++) {
			int jStart = Math.max(0, i - maxTokenLength);
			for (int j = jStart; j < i; j++) {
				if (bestScore[j] == Float.NEGATIVE_INFINITY) {
					continue;
				}
				String sub = text.substring(j, i);
				Integer id = vocab.get(sub);
				if (id != null) {
					float candidate = bestScore[j] + scores[id];
					if (candidate > bestScore[i]) {
						bestScore[i] = candidate;
						bestPrev[i] = j;
						bestTokenId[i] = id;
					}
				}
			}

			// If unreachable via vocab, use byte fallback for one codepoint
			if (bestScore[i] == Float.NEGATIVE_INFINITY) {
				if (i >= 1 && bestScore[i - 1] > Float.NEGATIVE_INFINITY
						&& !Character.isHighSurrogate(text.charAt(i - 1))) {
					// BMP character (single char)
					bestScore[i] = bestScore[i - 1] - 100.0f;
					bestPrev[i] = i - 1;
					bestTokenId[i] = -1;
				} else if (i >= 2 && bestScore[i - 2] > Float.NEGATIVE_INFINITY
						&& Character.isHighSurrogate(text.charAt(i - 2))
						&& Character.isLowSurrogate(text.charAt(i - 1))) {
					// Supplementary character (surrogate pair)
					bestScore[i] = bestScore[i - 2] - 100.0f;
					bestPrev[i] = i - 2;
					bestTokenId[i] = -1;
				}
			}
		}

		// Backtrack to recover optimal token sequence
		if (bestScore[len] == Float.NEGATIVE_INFINITY) {
			encodeByteFallback(text, tokenIds);
			return;
		}

		List<int[]> path = new ArrayList<>();
		int pos = len;
		while (pos > 0) {
			path.add(new int[]{bestPrev[pos], pos, bestTokenId[pos]});
			pos = bestPrev[pos];
		}
		Collections.reverse(path);

		for (int[] seg : path) {
			if (seg[2] >= 0) {
				tokenIds.add(seg[2]);
			} else {
				encodeByteFallback(text.substring(seg[0], seg[1]), tokenIds);
			}
		}
	}

	private void encodeByteFallback(String text, List<Integer> tokenIds) {
		byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
		for (byte b : bytes) {
			int id = byteFallbackIds[b & 0xFF];
			if (id >= 0) {
				tokenIds.add(id);
			}
		}
	}

	private static void flushBytes(ByteArrayOutputStream pending, StringBuilder sb) {
		if (pending.size() > 0) {
			sb.append(pending.toString(StandardCharsets.UTF_8));
			pending.reset();
		}
	}

	public static class Builder {

		private Map<String, Integer> vocab = new LinkedHashMap<>();
		private float[] scores = new float[0];
		private int unkId = 0;
		private final List<String> addedTokens = new ArrayList<>();
		private int defaultMaxLength = 8192;

		public Builder vocab(Map<String, Integer> vocab) {
			this.vocab = vocab;
			return this;
		}

		public Builder scores(float[] scores) {
			this.scores = scores;
			return this;
		}

		public Builder unkId(int unkId) {
			this.unkId = unkId;
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

		public UnigramTokenizer build() {
			return new UnigramTokenizer(this);
		}
	}
}
