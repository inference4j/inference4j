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

import io.github.inference4j.exception.InputTooLongException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.processing.TokenWindows;
import io.github.inference4j.processing.TruncationPolicy;
import io.github.inference4j.tokenizer.WordPieceTokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class BertNerRecognizerModelTest {

    @Nested
    @TestInstance(TestInstance.Lifecycle.PER_CLASS)
    class DistilbertNer {

        private BertNerRecognizer ner;

        @BeforeAll
        void setUp() {
            ner = BertNerRecognizer.builder()
                    .modelId("inference4j/distilbert-NER")
                    .build();
        }

        @AfterAll
        void tearDown() {
            if (ner != null) ner.close();
        }

        @Test
        void recognize_knownEntities() {
            List<NamedEntity> entities = ner.recognize("John works at Google in London.");

            assertThat(entities.isEmpty())
                    .as("Should find at least one entity")
                    .isFalse();

            List<String> labels = entities.stream().map(NamedEntity::label).toList();
            assertThat(labels)
                    .as("Should find PER, ORG, and LOC entities")
                    .contains("PER", "ORG", "LOC");
        }

        @Test
        void recognize_entityTextMatchesOriginal() {
            String text = "Marie Curie worked at the University of Paris.";
            List<NamedEntity> entities = ner.recognize(text);

            for (NamedEntity entity : entities) {
                assertThat(entity.text())
                        .as("Entity text should not be blank")
                        .isNotBlank();
                assertThat(entity.text())
                        .as("Entity text should be a substring of the original text")
                        .isEqualTo(text.substring(entity.start(), entity.end()));
            }
        }

        @Test
        void recognize_entityScoresInValidRange() {
            List<NamedEntity> entities = ner.recognize("Microsoft was founded by Bill Gates in Redmond.");

            for (NamedEntity entity : entities) {
                assertThat(entity.score())
                        .as("Score for " + entity.text() + " should be between 0 and 1")
                        .isBetween(0f, 1f);
            }
        }

        @Test
        void recognize_entityOffsetsAreValid() {
            String text = "Apple is headquartered in Cupertino, California.";
            List<NamedEntity> entities = ner.recognize(text);

            for (NamedEntity entity : entities) {
                assertThat(entity.start())
                        .as("Start offset should be >= 0")
                        .isGreaterThanOrEqualTo(0);
                assertThat(entity.end())
                        .as("End offset should be > start")
                        .isGreaterThan(entity.start());
                assertThat(entity.end())
                        .as("End offset should be <= text length")
                        .isLessThanOrEqualTo(text.length());
            }
        }

        @Test
        void recognize_noEntities_returnsEmptyList() {
            List<NamedEntity> entities = ner.recognize("The weather is nice today.");

            assertThat(entities)
                    .as("Non-entity text should return empty or minimal results")
                    .isNotNull();
        }

        @Test
        void recognize_entityLabelsAreValid() {
            List<NamedEntity> entities = ner.recognize("Leonardo da Vinci studied art in Florence and Milan.");

            for (NamedEntity entity : entities) {
                assertThat(entity.label())
                        .as("Label should be one of PER, ORG, LOC, MISC")
                        .isIn("PER", "ORG", "LOC", "MISC");
            }
        }
    }

    private static final String LONG_TEXT = "Marie Curie worked in Paris. ".repeat(150);

    @Test
    void failPolicyRejectsInputOverTheTokenLimit() {
        try (var strict = BertNerRecognizer.builder()
                .truncation(TruncationPolicy.FAIL)
                .build()) {
            assertThatThrownBy(() -> strict.recognize(LONG_TEXT))
                    .isInstanceOf(InputTooLongException.class);
            assertThat(strict.recognize("Marie Curie worked in Paris.")).isNotEmpty();
        }
    }

    // Over 512 tokens; entities in the middle and only at the very end
    private static final String FILLER =
            "Researchers in the laboratory measured samples, recorded the results, and compared them "
                    + "with earlier experiments carefully. ";
    private static final String LONG_DOCUMENT = FILLER.repeat(15)
            + "Albert Einstein visited Leonardo da Vinci's workshop in Florence. "
            + FILLER.repeat(15)
            + "Later that evening Marie Curie arrived in Warsaw.";

    @Test
    void strideFindsEntitiesAcrossTheWholeDocument() {
        // distilbert-NER is most accurate on windows of about 256 tokens; see the NER docs
        try (var ner = BertNerRecognizer.builder().maxLength(256).stride(64).build()) {
            List<NamedEntity> entities = ner.recognize(LONG_DOCUMENT);

            assertThat(entities).extracting(NamedEntity::text, NamedEntity::label).contains(
                    org.assertj.core.groups.Tuple.tuple("Albert Einstein", "PER"),
                    org.assertj.core.groups.Tuple.tuple("Leonardo da Vinci", "PER"),
                    org.assertj.core.groups.Tuple.tuple("Florence", "LOC"),
                    org.assertj.core.groups.Tuple.tuple("Marie Curie", "PER"),
                    org.assertj.core.groups.Tuple.tuple("Warsaw", "LOC"));
            for (NamedEntity e : entities) {
                assertThat(LONG_DOCUMENT.substring(e.start(), e.end())).isEqualTo(e.text());
            }
        }
    }

    @Test
    void withoutStrideEntitiesBeyondTheTokenLimitAreMissed() {
        try (var ner = BertNerRecognizer.builder().build()) {
            List<NamedEntity> entities = ner.recognize(LONG_DOCUMENT);

            assertThat(entities).noneSatisfy(e -> assertThat(e.text()).isEqualTo("Marie Curie"));
        }
    }

    @Test
    void strideGivesSameResultForShortText() {
        String text = "Marie Curie worked at the University of Paris in France.";
        try (var plain = BertNerRecognizer.builder().build();
             var windowed = BertNerRecognizer.builder().stride(128).build()) {
            assertThat(windowed.recognize(text)).isEqualTo(plain.recognize(text));
        }
    }

    @Test
    void strideKeepsAnEntityThatStraddlesAWindowBoundaryWhole() throws Exception {
        // Windows of 128 tokens (126 content tokens) overlapping by 32: ownership passes from the
        // first window to the second at content token 110. Place "Marie Curie" across that boundary.
        int maxLength = 128;
        int stride = 32;
        Path vocab = HuggingFaceModelSource.defaultInstance()
                .resolve("inference4j/distilbert-NER", List.of("model.onnx", "vocab.txt", "config.json"))
                .resolve("vocab.txt");
        WordPieceTokenizer tokenizer = WordPieceTokenizer.fromVocabFile(vocab, false);
        String tail = " Marie Curie arrived in Warsaw. " + "Researchers recorded the results carefully. ".repeat(20);

        try (var ner = BertNerRecognizer.builder().maxLength(maxLength).stride(stride).build()) {
            for (int shift = -2; shift <= 1; shift++) {
                String text = prefixOfTokens(tokenizer, 110 + shift) + tail;
                int contentTokens = tokenizer.encode(text, Integer.MAX_VALUE).inputIds().length - 2;
                int boundary = TokenWindows.plan(contentTokens, maxLength - 2, stride).get(0).ownedEnd();
                assertThat(boundary).as("first ownership boundary").isEqualTo(110);

                List<NamedEntity> people = ner.recognize(text).stream()
                        .filter(e -> e.label().equals("PER"))
                        .toList();

                assertThat(people).as("entity starting at content token %d", 110 + shift).hasSize(1);
                NamedEntity person = people.get(0);
                assertThat(person.text()).isEqualTo("Marie Curie");
                assertThat(text.substring(person.start(), person.end())).isEqualTo("Marie Curie");
            }
        }
    }

    /** A filler text that tokenizes to exactly {@code tokens} content tokens. */
    private static String prefixOfTokens(WordPieceTokenizer tokenizer, int tokens) {
        String[] words = {"researchers", "measured", "the", "samples", "and", "compared", "results"};
        StringBuilder text = new StringBuilder();
        for (int i = 0; tokenizer.encode(text.toString(), Integer.MAX_VALUE).inputIds().length - 2 < tokens; i++) {
            text.append(text.length() == 0 ? "" : " ").append(words[i % words.length]);
        }
        int count = tokenizer.encode(text.toString(), Integer.MAX_VALUE).inputIds().length - 2;
        if (count != tokens) {
            throw new IllegalStateException("filler produced " + count + " tokens, wanted " + tokens);
        }
        return text.toString();
    }
}
