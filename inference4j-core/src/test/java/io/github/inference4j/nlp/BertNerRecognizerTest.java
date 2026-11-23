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

import io.github.inference4j.TestModels;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.text.ModelConfig;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class BertNerRecognizerTest {

    @TempDir
    Path dir;

    // IOB2 labels matching CoNLL-2003 NER models
    private static final ModelConfig NER_CONFIG = ModelConfig.of(
            Map.of(0, "O", 1, "B-PER", 2, "I-PER", 3, "B-ORG", 4, "I-ORG",
                    5, "B-LOC", 6, "I-LOC", 7, "B-MISC", 8, "I-MISC"),
            null
    );

    // --- aggregateEntities tests ---

    @Test
    void aggregateEntities_singleWordEntity() {
        // "John works in London" → word 0=John(B-PER), 1=works(O), 2=in(O), 3=London(B-LOC)
        String text = "John works in London";
        String[] labels = {"B-PER", "O", "O", "B-LOC"};
        float[] scores = {0.98f, 0.99f, 0.99f, 0.97f};
        int[] wordIds = {0, 1, 2, 3};

        List<NamedEntity> entities = BertNerRecognizer.aggregateEntities(
                text, labels, scores, wordIds, 4);

        assertThat(entities).hasSize(2);
        assertThat(entities.get(0).text()).isEqualTo("John");
        assertThat(entities.get(0).label()).isEqualTo("PER");
        assertThat(entities.get(0).start()).isEqualTo(0);
        assertThat(entities.get(0).end()).isEqualTo(4);
        assertThat(entities.get(0).score()).isCloseTo(0.98f, within(1e-5f));
        assertThat(entities.get(1).text()).isEqualTo("London");
        assertThat(entities.get(1).label()).isEqualTo("LOC");
    }

    @Test
    void aggregateEntities_multiWordEntity() {
        // "New York is great" → word 0=New(B-LOC), 1=York(I-LOC), 2=is(O), 3=great(O)
        String text = "New York is great";
        String[] labels = {"B-LOC", "I-LOC", "O", "O"};
        float[] scores = {0.95f, 0.93f, 0.99f, 0.99f};
        int[] wordIds = {0, 1, 2, 3};

        List<NamedEntity> entities = BertNerRecognizer.aggregateEntities(
                text, labels, scores, wordIds, 4);

        assertThat(entities).hasSize(1);
        assertThat(entities.get(0).text()).isEqualTo("New York");
        assertThat(entities.get(0).label()).isEqualTo("LOC");
        assertThat(entities.get(0).start()).isEqualTo(0);
        assertThat(entities.get(0).end()).isEqualTo(8);
        assertThat(entities.get(0).score()).isCloseTo((0.95f + 0.93f) / 2, within(1e-5f));
    }

    @Test
    void aggregateEntities_noEntities() {
        String text = "Nothing to see here";
        String[] labels = {"O", "O", "O", "O"};
        float[] scores = {0.99f, 0.99f, 0.99f, 0.99f};
        int[] wordIds = {0, 1, 2, 3};

        List<NamedEntity> entities = BertNerRecognizer.aggregateEntities(
                text, labels, scores, wordIds, 4);

        assertThat(entities).isEmpty();
    }

    @Test
    void aggregateEntities_consecutiveEntitiesDifferentTypes() {
        // "John Google" → word 0=John(B-PER), 1=Google(B-ORG)
        String text = "John Google";
        String[] labels = {"B-PER", "B-ORG"};
        float[] scores = {0.95f, 0.92f};
        int[] wordIds = {0, 1};

        List<NamedEntity> entities = BertNerRecognizer.aggregateEntities(
                text, labels, scores, wordIds, 2);

        assertThat(entities).hasSize(2);
        assertThat(entities.get(0).text()).isEqualTo("John");
        assertThat(entities.get(0).label()).isEqualTo("PER");
        assertThat(entities.get(1).text()).isEqualTo("Google");
        assertThat(entities.get(1).label()).isEqualTo("ORG");
    }

    @Test
    void aggregateEntities_specialTokensSkipped() {
        // With [CLS] and [SEP] tokens (wordId = -1)
        String text = "John";
        String[] labels = {"O", "B-PER", "O"};
        float[] scores = {0.5f, 0.95f, 0.5f};
        int[] wordIds = {-1, 0, -1};

        List<NamedEntity> entities = BertNerRecognizer.aggregateEntities(
                text, labels, scores, wordIds, 3);

        assertThat(entities).hasSize(1);
        assertThat(entities.get(0).text()).isEqualTo("John");
        assertThat(entities.get(0).label()).isEqualTo("PER");
    }

    @Test
    void aggregateEntities_subwordAlignment() {
        // "Unbelievable" tokenized as "Un" + "##believable" → both get wordId=0
        // But at word level, only the first subtoken's label is used
        String text = "Unbelievable";
        String[] labels = {"O", "B-MISC", "I-MISC", "O"};  // [CLS], Un, ##believable, [SEP]
        float[] scores = {0.5f, 0.90f, 0.88f, 0.5f};
        int[] wordIds = {-1, 0, 0, -1};  // both subtokens map to word 0

        List<NamedEntity> entities = BertNerRecognizer.aggregateEntities(
                text, labels, scores, wordIds, 4);

        assertThat(entities).hasSize(1);
        assertThat(entities.get(0).text()).isEqualTo("Unbelievable");
        assertThat(entities.get(0).label()).isEqualTo("MISC");
        assertThat(entities.get(0).score()).isCloseTo(0.90f, within(1e-5f));
    }

    @Test
    void aggregateEntities_punctuationAsSeparateWord() {
        // "London." → word 0=London(B-LOC), word 1=.(O)
        String text = "London.";
        String[] labels = {"O", "B-LOC", "O", "O"};  // [CLS], London, ., [SEP]
        float[] scores = {0.5f, 0.95f, 0.99f, 0.5f};
        int[] wordIds = {-1, 0, 1, -1};

        List<NamedEntity> entities = BertNerRecognizer.aggregateEntities(
                text, labels, scores, wordIds, 4);

        assertThat(entities).hasSize(1);
        assertThat(entities.get(0).text()).isEqualTo("London");
        assertThat(entities.get(0).label()).isEqualTo("LOC");
        assertThat(entities.get(0).start()).isEqualTo(0);
        assertThat(entities.get(0).end()).isEqualTo(6);
    }

    @Test
    void aggregateEntities_iWithoutB_ignored() {
        // I-PER without preceding B-PER — should be ignored
        String text = "John";
        String[] labels = {"I-PER"};
        float[] scores = {0.95f};
        int[] wordIds = {0};

        List<NamedEntity> entities = BertNerRecognizer.aggregateEntities(
                text, labels, scores, wordIds, 1);

        assertThat(entities).isEmpty();
    }

    // --- splitIntoWords tests ---

    @Test
    void splitIntoWords_simpleText() {
        List<BertNerRecognizer.WordSpan> spans = BertNerRecognizer.splitIntoWords("Hello world");
        assertThat(spans).hasSize(2);
        assertThat(spans.get(0).start()).isEqualTo(0);
        assertThat(spans.get(0).end()).isEqualTo(5);
        assertThat(spans.get(1).start()).isEqualTo(6);
        assertThat(spans.get(1).end()).isEqualTo(11);
    }

    @Test
    void splitIntoWords_withPunctuation() {
        List<BertNerRecognizer.WordSpan> spans = BertNerRecognizer.splitIntoWords("Hello, world!");
        assertThat(spans).hasSize(4); // "Hello", ",", "world", "!"
        assertThat(spans.get(0).start()).isEqualTo(0);
        assertThat(spans.get(0).end()).isEqualTo(5);
        assertThat(spans.get(1).start()).isEqualTo(5);
        assertThat(spans.get(1).end()).isEqualTo(6);
    }

    @Test
    void splitIntoWordsSplitsAsciiSymbolsLikeTheTokenizer() {
        // "$" and "=" are words of their own for the tokenizer, so they must be here too,
        // or every later word index points at the wrong characters
        List<BertNerRecognizer.WordSpan> spans = BertNerRecognizer.splitIntoWords("Paid $5 to Marie=Curie");

        assertThat(spans).extracting(BertNerRecognizer.WordSpan::start)
                .containsExactly(0, 5, 6, 8, 11, 16, 17);
    }

    @Test
    void splitIntoWordsSkipsCharactersTheTokenizerDrops() {
        // A lone control character is no word for the tokenizer, so it must not be one here
        List<BertNerRecognizer.WordSpan> spans = BertNerRecognizer.splitIntoWords("Marie \u0085 Curie");

        assertThat(spans).extracting(BertNerRecognizer.WordSpan::start).containsExactly(0, 8);
    }

    @Test
    void splitIntoWords_multipleSpaces() {
        List<BertNerRecognizer.WordSpan> spans = BertNerRecognizer.splitIntoWords("Hello  world");
        assertThat(spans).hasSize(2);
    }

    // --- Builder validation ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThatThrownBy(() ->
                BertNerRecognizer.builder()
                        .tokenizer(tokenizer)
                        .config(NER_CONFIG)
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_missingTokenizer_throws() {
        ModelSource source = TestModels.model(dir, "tokens-onehot");
        assertThatThrownBy(() ->
                BertNerRecognizer.builder()
                        .modelSource(source)
                        .config(NER_CONFIG)
                        .build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("vocab.txt");
    }

    @Test
    void builder_missingConfig_throws() {
        ModelSource source = TestModels.model(dir, "tokens-onehot");
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThatThrownBy(() ->
                BertNerRecognizer.builder()
                        .modelSource(source)
                        .tokenizer(tokenizer)
                        .build())
                .isInstanceOf(ModelSourceException.class)
                .hasMessageContaining("config.json");
    }

    // --- Inference flow ---
    // The fixture model gives each token a logit of 10 at the index equal to its token id, so the token ids
    // below are the label indices: O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6.

    @Test
    void recognize_endToEnd() {
        Tokenizer tokenizer = mock(Tokenizer.class);

        // "John works at Google" → 4 words + [CLS] + [SEP] = 6 tokens
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{0, 1, 0, 0, 3, 0},
                        new long[]{1, 1, 1, 1, 1, 1},
                        new long[]{0, 0, 0, 0, 0, 0},
                        new int[]{-1, 0, 1, 2, 3, -1}
                ));

        try (BertNerRecognizer ner = BertNerRecognizer.builder()
                .modelSource(TestModels.model(dir, "tokens-onehot-types"))
                .tokenizer(tokenizer)
                .config(NER_CONFIG)
                .build()) {
            List<NamedEntity> entities = ner.recognize("John works at Google");

            assertThat(entities).hasSize(2);
            assertThat(entities.get(0).text()).isEqualTo("John");
            assertThat(entities.get(0).label()).isEqualTo("PER");
            assertThat(entities.get(0).start()).isEqualTo(0);
            assertThat(entities.get(0).end()).isEqualTo(4);
            assertThat(entities.get(0).score()).isGreaterThan(0.99f);

            assertThat(entities.get(1).text()).isEqualTo("Google");
            assertThat(entities.get(1).label()).isEqualTo("ORG");
            assertThat(entities.get(1).start()).isEqualTo(14);
            assertThat(entities.get(1).end()).isEqualTo(20);
        }
    }

    @Test
    void recognize_multiWordEntity() {
        Tokenizer tokenizer = mock(Tokenizer.class);

        // "New York" → 2 words + [CLS] + [SEP] = 4 tokens: B-LOC, I-LOC
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{0, 5, 6, 0},
                        new long[]{1, 1, 1, 1},
                        new long[]{0, 0, 0, 0},
                        new int[]{-1, 0, 1, -1}
                ));

        try (BertNerRecognizer ner = recognizer(tokenizer)) {
            List<NamedEntity> entities = ner.recognize("New York");

            assertThat(entities).hasSize(1);
            assertThat(entities.get(0).text()).isEqualTo("New York");
            assertThat(entities.get(0).label()).isEqualTo("LOC");
        }
    }

    @Test
    void recognize_noEntities_returnsEmptyList() {
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{0, 0, 0},
                        new long[]{1, 1, 1},
                        new long[]{0, 0, 0},
                        new int[]{-1, 0, -1}
                ));

        try (BertNerRecognizer ner = recognizer(tokenizer)) {
            assertThat(ner.recognize("this")).isEmpty();
        }
    }

    @Test
    void strideLabelsEveryTokenAcrossWindows() {
        // 8 words, 10 tokens with [CLS] and [SEP]: longer than maxLength 6, so it runs in windows of 4 tokens
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{0, 1, 0, 0, 3, 0, 0, 5, 6, 0},
                        new long[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                        new long[10],
                        new int[]{-1, 0, 1, 2, 3, 4, 5, 6, 7, -1}
                ));

        try (BertNerRecognizer ner = BertNerRecognizer.builder()
                .modelSource(TestModels.model(dir, "tokens-onehot"))
                .tokenizer(tokenizer)
                .config(NER_CONFIG)
                .maxLength(6)
                .stride(2)
                .build()) {
            List<NamedEntity> entities = ner.recognize("Alice met the Acme team in New York");

            assertThat(entities).extracting(NamedEntity::text, NamedEntity::label).containsExactly(
                    tuple("Alice", "PER"), tuple("Acme", "ORG"), tuple("New York", "LOC"));
        }
    }

    // --- Close ---

    @Test
    void closedRecognizerRejectsRecognize() {
        Tokenizer tokenizer = mock(Tokenizer.class);
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(new long[]{0, 0, 0}, new long[]{1, 1, 1}, new long[]{0, 0, 0},
                        new int[]{-1, 0, -1}));
        BertNerRecognizer ner = recognizer(tokenizer);

        ner.close();

        assertThatThrownBy(() -> ner.recognize("this")).isInstanceOf(IllegalStateException.class);
    }

    private BertNerRecognizer recognizer(Tokenizer tokenizer) {
        return BertNerRecognizer.builder()
                .modelSource(TestModels.model(dir, "tokens-onehot"))
                .tokenizer(tokenizer)
                .config(NER_CONFIG)
                .build();
    }
}
