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
import io.github.inference4j.processing.TruncationPolicy;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.tuple;
import static org.assertj.core.data.Offset.offset;

/**
 * Expected entities and scores are the {@code gliner} Python package's (0.2.29) on the same ONNX models, which
 * inference4j matches to within 1e-6.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class GlinerEntityRecognizerModelTest {

    private static final String INVOICE = "Alice Johnson paid invoice INV-2024-0042 to Acme Corp on 3 March.";
    private static final List<String> LABELS = List.of("person", "organization", "invoice id", "date");
    private static final String FILLER = IntStream.range(0, 385).mapToObj(i -> "filler" + i)
            .collect(Collectors.joining(" "));

    private GlinerEntityRecognizer recognizer;

    @BeforeAll
    void setUp() {
        recognizer = GlinerEntityRecognizer.builder().build();
    }

    @AfterAll
    void tearDown() {
        if (recognizer != null) {
            recognizer.close();
        }
    }

    @Test
    void findsEntitiesOfTheLabelsGivenAtCallTime() {
        List<NamedEntity> entities = recognizer.recognize(INVOICE, LABELS);

        assertThat(entities).extracting(NamedEntity::text, NamedEntity::label, NamedEntity::start, NamedEntity::end)
                .containsExactly(
                        tuple("Alice Johnson", "person", 0, 13),
                        tuple("INV-2024-0042", "invoice id", 27, 40),
                        tuple("Acme Corp", "organization", 44, 53),
                        tuple("3 March", "date", 57, 64));
        assertThat(entities).extracting(NamedEntity::score).satisfiesExactly(
                s -> assertThat(s).isCloseTo(0.9836f, offset(1e-3f)),
                s -> assertThat(s).isCloseTo(0.8685f, offset(1e-3f)),
                s -> assertThat(s).isCloseTo(0.9660f, offset(1e-3f)),
                s -> assertThat(s).isCloseTo(0.9326f, offset(1e-3f)));
    }

    @Test
    void recognizeWithoutLabelsUsesTheDefaultLabels() {
        NamedEntityRecognizer ner = recognizer;   // person, organization, location

        assertThat(ner.recognize("Marie Curie worked at the Radium Institute in Paris."))
                .extracting(NamedEntity::text, NamedEntity::label)
                .containsExactly(
                        tuple("Marie Curie", "person"),
                        tuple("Radium Institute", "organization"),
                        tuple("Paris", "location"));
    }

    @Test
    void longTextIsTruncatedOrReadInWindows() {
        String text = FILLER + " Alice Johnson joined Acme Corp on 3 March.";
        List<String> labels = List.of("person", "organization", "date");

        assertThat(recognizer.recognize(text, labels)).isEmpty();   // the entities lie past the 384-word limit
        try (var windowed = GlinerEntityRecognizer.builder().stride(64).build()) {
            assertThat(windowed.recognize(text, labels)).extracting(NamedEntity::text, NamedEntity::label)
                    .contains(tuple("Alice Johnson", "person"), tuple("3 March", "date"));
        }
        try (var strict = GlinerEntityRecognizer.builder().truncation(TruncationPolicy.FAIL).build()) {
            assertThatThrownBy(() -> strict.recognize(text, labels)).isInstanceOf(InputTooLongException.class);
        }
    }

    @Test
    void piiModelFindsPersonalData() {
        String message = "Hi, I'm Alice Johnson. Reach me at alice.johnson@example.com or +1 415 555 0132. "
                + "My card is 4111 1111 1111 1111 and I live at 42 Elm Street, Springfield.";
        try (var pii = GlinerEntityRecognizer.pii().threshold(0.3f).build()) {
            assertThat(pii.recognize(message)).extracting(NamedEntity::text, NamedEntity::label)
                    .contains(
                            tuple("alice.johnson@example.com", "email"),
                            tuple("+1 415 555 0132", "phone number"),
                            tuple("4111 1111 1111 1111", "credit card number"),
                            tuple("42 Elm Street, Springfield", "address"));

            List<NamedEntity> invoice = pii.recognize(INVOICE, LABELS);
            assertThat(invoice).extracting(NamedEntity::label)
                    .containsExactly("person", "invoice id", "organization", "date");
            assertThat(invoice.get(1).score()).isCloseTo(0.9846f, offset(1e-3f));
        }
    }

    @Test
    void rejectsAThresholdOutsideTheUnitInterval() {
        assertThatThrownBy(() -> GlinerEntityRecognizer.builder().threshold(1f).build())
                .isInstanceOf(IllegalArgumentException.class);
    }
}
