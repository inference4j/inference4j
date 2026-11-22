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

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Checks each published MarianMT model against the original: the expected texts are the greedy output of the
 * Helsinki-NLP PyTorch model with {@code MarianTokenizer} ({@code num_beams=1}, {@code do_sample=False}).
 * A tokenizer whose ids differ from the model's (#214) fails here, where a non-blank check passes.
 */
class MarianTranslatorReferenceModelTest {

    static Stream<Arguments> referenceTranslations() {
        return Stream.of(
                Arguments.of("inference4j/opus-mt-en-de",
                        "The oldest official map of Munich brings captivating stories to light.",
                        "Die älteste offizielle Karte von München bringt fesselnde Geschichten ans Licht."),
                Arguments.of("inference4j/opus-mt-en-de",
                        "Hello, how are you?",
                        "Hallo, wie geht's?"),
                Arguments.of("inference4j/opus-mt-en-de",
                        "The meeting starts at 9:30 a.m. on March 3rd, 2025, in room 42.",
                        "Das Treffen beginnt um 9:30 Uhr am 3. März 2025 in Raum 42."),
                Arguments.of("inference4j/opus-mt-en-de",
                        "\"I don't think so,\" he said, 'but maybe you're right.'",
                        "\"Ich glaube nicht\", sagte er, \"aber vielleicht hast du recht.\""),
                Arguments.of("inference4j/opus-mt-en-de",
                        "The weather is beautiful today, isn't it?",
                        "Das Wetter ist heute schön, nicht wahr?"),
                Arguments.of("inference4j/opus-mt-en-fr",
                        "The oldest official map of Munich brings captivating stories to light.",
                        "La plus ancienne carte officielle de Munich met en lumière des histoires captivantes."),
                Arguments.of("inference4j/opus-mt-en-fr",
                        "Hello, how are you?",
                        "Bonjour, comment allez-vous ?"),
                Arguments.of("inference4j/opus-mt-en-fr",
                        "The meeting starts at 9:30 a.m. on March 3rd, 2025, in room 42.",
                        "La réunion commence le 3 mars 2025 à 9 h 30 dans la salle 42."),
                Arguments.of("inference4j/opus-mt-en-fr",
                        "\"I don't think so,\" he said, 'but maybe you're right.'",
                        "\"Je ne pense pas,\" dit-il, \"mais peut-être que tu as raison.\""),
                Arguments.of("inference4j/opus-mt-en-fr",
                        "The weather is beautiful today, isn't it?",
                        "Le temps est beau aujourd'hui, n'est-ce pas ?"),
                Arguments.of("inference4j/opus-mt-en-es",
                        "The oldest official map of Munich brings captivating stories to light.",
                        "El mapa oficial más antiguo de Múnich trae historias cautivadoras a la luz."),
                Arguments.of("inference4j/opus-mt-en-es",
                        "Hello, how are you?",
                        "Hola, ¿cómo estás?"),
                Arguments.of("inference4j/opus-mt-en-es",
                        "The meeting starts at 9:30 a.m. on March 3rd, 2025, in room 42.",
                        "La reunión comienza a las 9:30 a.m. del 3 de marzo de 2025, en la sala 42."),
                Arguments.of("inference4j/opus-mt-en-es",
                        "\"I don't think so,\" he said, 'but maybe you're right.'",
                        "\"No lo creo\", dijo, \"pero tal vez tengas razón\"."),
                Arguments.of("inference4j/opus-mt-en-es",
                        "The weather is beautiful today, isn't it?",
                        "El tiempo es hermoso hoy, ¿no?")
        );
    }

    @ParameterizedTest(name = "{0}: {1}")
    @MethodSource("referenceTranslations")
    void translatesLikeTheOriginalModel(String modelId, String english, String expected) throws Exception {
        try (MarianTranslator translator = MarianTranslator.builder().modelId(modelId).build()) {
            assertThat(translator.translate(english)).isEqualTo(expected);
        }
    }
}
