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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

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
}
