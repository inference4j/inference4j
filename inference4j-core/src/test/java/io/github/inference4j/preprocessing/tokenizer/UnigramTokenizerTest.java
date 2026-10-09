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

package io.github.inference4j.preprocessing.tokenizer;

import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.TokenizerJsonParser;
import io.github.inference4j.tokenizer.UnigramTokenizer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import static org.assertj.core.api.Assertions.assertThat;

class UnigramTokenizerTest {

	private static UnigramTokenizer tokenizer;

	@BeforeAll
	static void setUp() {
		Path tokenizerJson = Path.of(Objects.requireNonNull(
				UnigramTokenizerTest.class.getResource("/test-unigram-tokenizer.json")).getPath());
		tokenizer = TokenizerJsonParser.parseUnigram(tokenizerJson).build();
	}

	@Test
	void encode_viterbiOptimalPath() {
		// "hello" → "▁hello" → Viterbi picks single token (score -5.0)
		// over ▁he(-4.5) + lo(-5.0) = -9.5
		EncodedInput result = tokenizer.encode("hello", 512);
		assertThat(result.inputIds()[0]).as("▁hello should be token 12").isEqualTo(12);
		assertThat(result.inputIds().length).as("single merged token via Viterbi").isEqualTo(1);
	}

	@Test
	void encode_viterbiMultiTokenPath() {
		// "helo" → "▁helo" → not in vocab, Viterbi picks ▁he(10) + lo(11) = -9.5
		// which beats ▁h(9) + e(6) + l(7) + o(8) = -14.5
		EncodedInput result = tokenizer.encode("helo", 512);
		assertThat(result.inputIds()).as("Viterbi should pick ▁he + lo")
				.isEqualTo(new long[]{10, 11});
	}

	@Test
	void encode_prependsSpacePrefix() {
		// "hello" → "▁hello" → token 12
		EncodedInput result = tokenizer.encode("hello", 512);
		assertThat(result.inputIds()[0]).as("▁hello should be token 12").isEqualTo(12);
	}

	@Test
	void encode_handlesSpacesBetweenWords() {
		// "hello world" → "▁hello▁world" → tokens 12, 16
		EncodedInput result = tokenizer.encode("hello world", 512);
		assertThat(result.inputIds()).isEqualTo(new long[]{12, 16});
	}

	@Test
	void encode_addedTokensAtomic() {
		// "<start_of_turn>hello" → special token 297 + ▁hello(12)
		EncodedInput result = tokenizer.encode("<start_of_turn>hello", 512);
		assertThat(result.inputIds()[0]).as("<start_of_turn> should be token 297").isEqualTo(297);
		assertThat(result.inputIds()[1]).as("▁hello should be token 12").isEqualTo(12);
	}

	@Test
	void encode_byteFallbackForUnknownChars() {
		// "!" is not in the vocab → byte fallback
		// Input: "!" → "▁!" → ▁ (token 3) + byte fallback for '!' (0x21)
		// <0x21> is at ID 41 + 0x21 = 41 + 33 = 74
		EncodedInput result = tokenizer.encode("!", 512);
		assertThat(result.inputIds()[0]).as("▁ should be token 3").isEqualTo(3);
		assertThat(result.inputIds()[1]).as("! should be byte fallback <0x21> = 74").isEqualTo(74);
	}

	@Test
	void decode_reversesEncoding() {
		EncodedInput encoded = tokenizer.encode("hello world", 512);
		int[] ids = new int[encoded.inputIds().length];
		for (int i = 0; i < ids.length; i++) {
			ids[i] = (int) encoded.inputIds()[i];
		}
		String decoded = tokenizer.decode(ids);
		assertThat(decoded).isEqualTo("hello world");
	}

	@Test
	void decode_singleToken_streaming() {
		// Token 12 is "▁hello" → should decode to " hello" (▁ replaced with space)
		String result = tokenizer.decode(12);
		assertThat(result).isEqualTo(" hello");
	}

	@Test
	void decode_skipsSpecialTokens() {
		// <pad>(0) and <eos>(1) are added tokens → should be skipped
		String result = tokenizer.decode(new int[]{0, 12, 16, 1});
		assertThat(result).isEqualTo("hello world");
	}

	@Test
	void decode_byteFallbackTokens() {
		// <0x21> (token 74) is byte 0x21 = '!'
		String result = tokenizer.decode(new int[]{12, 74});
		// ▁hello + ! → "hello!"
		assertThat(result).isEqualTo("hello!");
	}

	@Test
	void encode_attentionMaskAllOnes() {
		EncodedInput result = tokenizer.encode("hello", 512);
		for (long mask : result.attentionMask()) {
			assertThat(mask).isEqualTo(1L);
		}
	}

	@Test
	void encode_tokenTypeIdsAllZeros() {
		EncodedInput result = tokenizer.encode("hello world", 512);
		for (long typeId : result.tokenTypeIds()) {
			assertThat(typeId).isEqualTo(0L);
		}
	}

	@Test
	void encode_truncatesToMaxLength() {
		EncodedInput result = tokenizer.encode("hello world", 1);
		assertThat(result.inputIds().length).isEqualTo(1);
	}

	@Test
	void encode_multipleAddedTokens() {
		// "<start_of_turn>hello<end_of_turn>" → 297, 12, 298
		EncodedInput result = tokenizer.encode("<start_of_turn>hello<end_of_turn>", 512);
		assertThat(result.inputIds()[0]).isEqualTo(297);
		assertThat(result.inputIds().length).isGreaterThanOrEqualTo(3);
		assertThat(result.inputIds()[result.inputIds().length - 1]).isEqualTo(298);
	}

	@Test
	void decode_multiByteFallback() {
		// UTF-8 encoding of 'é' is 0xC3 0xA9
		// <0xC3> = 41 + 0xC3 = 41 + 195 = 236
		// <0xA9> = 41 + 0xA9 = 41 + 169 = 210
		String result = tokenizer.decode(new int[]{236, 210});
		assertThat(result).as("Two byte-fallback tokens should decode to 'é'").isEqualTo("\u00e9");
	}

	@Test
	void encode_emptyString() {
		// Empty string produces no tokens (consistent with SentencePiece BPE behavior)
		EncodedInput result = tokenizer.encode("", 512);
		assertThat(result.inputIds().length).isEqualTo(0);
	}

	// --- Models without byte-fallback tokens (e.g. T5, punctuation restoration) ---
	// Expected IDs mirror SentencePiece: a run of unmapped characters becomes a single unk.

	private static UnigramTokenizer noByteFallbackTokenizer(int unkId) {
		Map<String, Integer> vocab = new LinkedHashMap<>();
		vocab.put("<unk>", 0);
		vocab.put("\u2581", 1);
		vocab.put("\u2581na", 2);
		vocab.put("ve", 3);
		vocab.put("\u2581text", 4);
		vocab.put("<extra_unk>", 5);
		float[] scores = {0f, -3f, -5f, -5f, -5f, 0f};
		return UnigramTokenizer.builder().vocab(vocab).scores(scores).unkId(unkId).build();
	}

	@Test
	void encodeEmptyStringWithBareSpaceMarkerInVocabProducesNoTokens() {
		EncodedInput result = noByteFallbackTokenizer(0).encode("", 512);
		assertThat(result.inputIds()).isEmpty();
	}

	@Test
	void encodeNoByteFallbackUnknownCharBecomesUnk() {
		EncodedInput result = noByteFallbackTokenizer(0).encode("na\u00efve", 512);
		assertThat(result.inputIds()).containsExactly(2L, 0L, 3L);
	}

	@Test
	void encodeNoByteFallbackRunOfUnknownCharsBecomesSingleUnk() {
		EncodedInput result = noByteFallbackTokenizer(0).encode("\u00ef\u00ef", 512);
		assertThat(result.inputIds()).containsExactly(1L, 0L);
	}

	@Test
	void encodeNoByteFallbackUnknownRunsSeparatedBySpaceEachGetUnk() {
		EncodedInput result = noByteFallbackTokenizer(0).encode("\u00ef \u00ef", 512);
		assertThat(result.inputIds()).containsExactly(1L, 0L, 1L, 0L);
	}

	@Test
	void encodeNoByteFallbackSupplementaryAndCjkCharsBecomeSingleUnk() {
		EncodedInput result = noByteFallbackTokenizer(0).encode("\u65e5\u672c\ud83d\ude00 text", 512);
		assertThat(result.inputIds()).containsExactly(1L, 0L, 4L);
	}

	@Test
	void encodeNoByteFallbackUsesConfiguredUnkId() {
		EncodedInput result = noByteFallbackTokenizer(5).encode("na\u00efve", 512);
		assertThat(result.inputIds()).containsExactly(2L, 5L, 3L);
	}

	// --- original length / truncation reporting ---

	@Test
	void encodeOverLimitReportsOriginalLength() {
		int fullLength = tokenizer.encode("hello world", 512).inputIds().length;

		EncodedInput result = tokenizer.encode("hello world", 1);

		assertThat(fullLength).isGreaterThan(1);
		assertThat(result.inputIds()).hasSize(1);
		assertThat(result.truncated()).isTrue();
		assertThat(result.originalLength()).isEqualTo(fullLength);
	}

	@Test
	void encodeWithinLimitIsNotTruncated() {
		EncodedInput result = tokenizer.encode("hello world", 512);

		assertThat(result.truncated()).isFalse();
		assertThat(result.originalLength()).isEqualTo(result.inputIds().length);
	}
}
