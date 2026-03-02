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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.text.ModelConfig;
import io.github.inference4j.tokenizer.EncodedInput;
import io.github.inference4j.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class BertNerRecognizerTest {

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
        InferenceSession session = mock(InferenceSession.class);
        assertThatThrownBy(() ->
                BertNerRecognizer.builder()
                        .session(session)
                        .config(NER_CONFIG)
                        .build())
                .isInstanceOf(IllegalStateException.class);
    }

    @Test
    void builder_missingConfig_throws() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);
        assertThatThrownBy(() ->
                BertNerRecognizer.builder()
                        .session(session)
                        .tokenizer(tokenizer)
                        .build())
                .isInstanceOf(IllegalStateException.class);
    }

    // --- Inference flow ---

    @Test
    void recognize_endToEnd_withMocks() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        // "John works at Google" → 4 words + [CLS] + [SEP] = 6 tokens
        // Word IDs: -1(CLS), 0(John), 1(works), 2(at), 3(Google), -1(SEP)
        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask", "token_type_ids"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{101, 2198, 2515, 2012, 8224, 102},
                        new long[]{1, 1, 1, 1, 1, 1},
                        new long[]{0, 0, 0, 0, 0, 0},
                        new int[]{-1, 0, 1, 2, 3, -1}
                ));

        // Model output: [1, 6, 9] logits — John=B-PER, works=O, at=O, Google=B-ORG
        // 9 labels: O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6, B-MISC=7, I-MISC=8
        float[] logits = new float[6 * 9];
        // Token 0 (CLS): O
        logits[0 * 9 + 0] = 10.0f;
        // Token 1 (John): B-PER
        logits[1 * 9 + 1] = 10.0f;
        // Token 2 (works): O
        logits[2 * 9 + 0] = 10.0f;
        // Token 3 (at): O
        logits[3 * 9 + 0] = 10.0f;
        // Token 4 (Google): B-ORG
        logits[4 * 9 + 3] = 10.0f;
        // Token 5 (SEP): O
        logits[5 * 9 + 0] = 10.0f;

        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(logits, new long[]{1, 6, 9})));

        BertNerRecognizer ner = BertNerRecognizer.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(NER_CONFIG)
                .build();

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

    @Test
    void recognize_multiWordEntity_withMocks() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        // "New York" → 2 words + [CLS] + [SEP] = 4 tokens
        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{101, 2739, 2088, 102},
                        new long[]{1, 1, 1, 1},
                        new long[]{0, 0, 0, 0},
                        new int[]{-1, 0, 1, -1}
                ));

        float[] logits = new float[4 * 9];
        logits[0 * 9 + 0] = 10.0f;  // CLS: O
        logits[1 * 9 + 5] = 10.0f;  // New: B-LOC
        logits[2 * 9 + 6] = 10.0f;  // York: I-LOC
        logits[3 * 9 + 0] = 10.0f;  // SEP: O

        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(logits, new long[]{1, 4, 9})));

        BertNerRecognizer ner = BertNerRecognizer.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(NER_CONFIG)
                .build();

        List<NamedEntity> entities = ner.recognize("New York");

        assertThat(entities).hasSize(1);
        assertThat(entities.get(0).text()).isEqualTo("New York");
        assertThat(entities.get(0).label()).isEqualTo("LOC");
    }

    @Test
    void recognize_noEntities_returnsEmptyList() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        when(session.inputNames()).thenReturn(Set.of("input_ids", "attention_mask"));
        when(tokenizer.encode(anyString(), anyInt())).thenReturn(
                new EncodedInput(
                        new long[]{101, 2023, 102},
                        new long[]{1, 1, 1},
                        new long[]{0, 0, 0},
                        new int[]{-1, 0, -1}
                ));

        float[] logits = new float[3 * 9];
        logits[0 * 9 + 0] = 10.0f;  // CLS: O
        logits[1 * 9 + 0] = 10.0f;  // "this": O
        logits[2 * 9 + 0] = 10.0f;  // SEP: O

        when(session.run(any())).thenReturn(
                Map.of("logits", Tensor.fromFloats(logits, new long[]{1, 3, 9})));

        BertNerRecognizer ner = BertNerRecognizer.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(NER_CONFIG)
                .build();

        List<NamedEntity> entities = ner.recognize("this");

        assertThat(entities).isEmpty();
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);
        Tokenizer tokenizer = mock(Tokenizer.class);

        BertNerRecognizer ner = BertNerRecognizer.builder()
                .session(session)
                .tokenizer(tokenizer)
                .config(NER_CONFIG)
                .build();

        ner.close();

        verify(session).close();
    }
}
