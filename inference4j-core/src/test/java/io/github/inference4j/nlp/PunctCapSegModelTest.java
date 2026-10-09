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

package io.github.inference4j.nlp;

import io.github.inference4j.InferenceContext;
import io.github.inference4j.Tensor;
import io.github.inference4j.tokenizer.UnigramTokenizer;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class PunctCapSegModelTest {

    // Post-punctuation labels, from the model's config.yaml
    private static final long NONE = 0;
    private static final long ACRONYM = 1;
    private static final long PERIOD = 2;
    private static final long COMMA = 3;
    private static final long QUESTION = 4;

    private static final String[] VOCAB = {
            "<unk>", "<s>", "</s>", "<pad>",
            "▁marie", "▁cur", "ie", "▁moved", "▁to", "▁paris",
            "▁she", "▁won", "▁the", "▁us", "▁prize", "▁why"};

    private static final UnigramTokenizer TOKENIZER = tokenizer();

    // --- normalize ---

    @Test
    void normalizeLowercasesAndCollapsesWhitespace() {
        assertThat(PunctCapSegModel.normalize("  MARIE   CURIE\tMOVED\n")).isEqualTo("marie curie moved");
    }

    @Test
    void normalizeDropsPunctuationButKeepsApostrophes() {
        assertThat(PunctCapSegModel.normalize("Hello, world! Don't stop... (ok?)"))
                .isEqualTo("hello world don't stop ok");
    }

    @Test
    void normalizeDropsUnicodePunctuation() {
        assertThat(PunctCapSegModel.normalize("“quoted” — dash…"))
                .isEqualTo("quoted dash");
    }

    @Test
    void normalizeBlankInputIsEmpty() {
        assertThat(PunctCapSegModel.normalize("  ...  ")).isEmpty();
    }

    // --- postProcess ---

    @Test
    void postProcessAppliesCasingAcrossSubwordsAndPunctuation() {
        // "▁marie ▁cur ie ▁moved ▁to ▁paris" → "Marie Curie moved to Paris."
        Prediction p = new Prediction("▁marie", "▁cur", "ie", "▁moved", "▁to", "▁paris");
        p.capitalize(0, 1).capitalize(1, 1).capitalize(5, 1);
        p.punctuate(5, PERIOD);

        assertThat(p.postProcess()).containsExactly("Marie Curie moved to Paris.");
    }

    @Test
    void postProcessSplitsSentencesOnBoundaries() {
        Prediction p = new Prediction("▁she", "▁moved", "▁she", "▁won");
        p.capitalize(0, 1).capitalize(2, 1);
        p.punctuate(1, PERIOD).endSentence(1);
        p.punctuate(3, PERIOD).endSentence(3);

        assertThat(p.postProcess()).containsExactly("She moved.", "She won.");
    }

    @Test
    void postProcessTrailingTextWithoutBoundaryIsStillReturned() {
        Prediction p = new Prediction("▁she", "▁won", "▁the", "▁prize");
        p.capitalize(0, 1);
        p.punctuate(1, PERIOD).endSentence(1);

        assertThat(p.postProcess()).containsExactly("She won.", "the prize");
    }

    @Test
    void postProcessAcronymGetsPeriodAfterEveryLetter() {
        Prediction p = new Prediction("▁the", "▁us", "▁prize");
        p.capitalize(1, 1).capitalize(1, 2);
        p.punctuate(1, ACRONYM);

        assertThat(p.postProcess()).containsExactly("the U.S. prize");
    }

    @Test
    void postProcessCommaAndQuestionMark() {
        Prediction p = new Prediction("▁why", "▁she", "▁won");
        p.capitalize(0, 1);
        p.punctuate(0, COMMA).punctuate(2, QUESTION).endSentence(2);

        assertThat(p.postProcess()).containsExactly("Why, she won?");
    }

    @Test
    void postProcessUnknownTokenStaysVisible() {
        Prediction p = new Prediction("▁she", "<unk>", "▁won");

        assertThat(p.postProcess()).containsExactly("she<unk> won");
    }

    @Test
    void postProcessNoTokensReturnsEmptyList() {
        assertThat(new Prediction().postProcess()).isEmpty();
    }

    // --- helpers ---

    private static UnigramTokenizer tokenizer() {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        for (int i = 0; i < VOCAB.length; i++) {
            vocab.put(VOCAB[i], i);
        }
        return UnigramTokenizer.builder().vocab(vocab).scores(new float[VOCAB.length]).build();
    }

    /** Builds fake model outputs for a token sequence, wrapped in BOS/EOS like the preprocessor does. */
    private static final class Prediction {

        private final long[] ids;
        private final long[] punctuation;
        private final boolean[] capitals;
        private final boolean[] sentenceEnds;

        Prediction(String... tokens) {
            int length = tokens.length + 2;
            ids = new long[length];
            ids[0] = 1;
            ids[length - 1] = 2;
            for (int i = 0; i < tokens.length; i++) {
                ids[i + 1] = List.of(VOCAB).indexOf(tokens[i]);
            }
            punctuation = new long[length];
            capitals = new boolean[length * 16];
            sentenceEnds = new boolean[length];
        }

        /** Upper-cases character {@code charIndex} of token {@code token}; index 0 is the ▁ marker. */
        Prediction capitalize(int token, int charIndex) {
            capitals[(token + 1) * 16 + charIndex] = true;
            return this;
        }

        Prediction punctuate(int token, long label) {
            punctuation[token + 1] = label;
            return this;
        }

        Prediction endSentence(int token) {
            sentenceEnds[token + 1] = true;
            return this;
        }

        List<String> postProcess() {
            int length = ids.length;
            Map<String, Tensor> preprocessed = Map.of("input_ids", Tensor.fromLongs(ids, new long[]{1, length}));
            Map<String, Tensor> outputs = Map.of(
                    "post_preds", Tensor.fromLongs(punctuation, new long[]{1, length}),
                    "cap_preds", Tensor.fromBooleans(capitals, new long[]{1, length, 16}),
                    "seg_preds", Tensor.fromBooleans(sentenceEnds, new long[]{1, length}));
            return PunctCapSegModel.postProcess(new InferenceContext<>("", preprocessed, outputs), TOKENIZER);
        }
    }
}
