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

package io.github.inference4j.preprocessing.audio;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class VocabularyTest {

    @Test
    void parseVocabJson_basicVocabulary() {
        String json = """
                {"a": 1, "b": 2, "c": 3}
                """;

        Map<Integer, String> map = io.github.inference4j.preprocessing.audio.Vocabulary.parseVocabJson(json);

        assertThat(map.size()).isEqualTo(3);
        assertThat(map.get(1)).isEqualTo("a");
        assertThat(map.get(2)).isEqualTo("b");
        assertThat(map.get(3)).isEqualTo("c");
    }

    @Test
    void parseVocabJson_specialCharacters() {
        String json = """
                {"<pad>": 0, "|": 4, "<unk>": 5, "'": 6}
                """;

        Map<Integer, String> map = io.github.inference4j.preprocessing.audio.Vocabulary.parseVocabJson(json);

        assertThat(map.get(0)).isEqualTo("<pad>");
        assertThat(map.get(4)).isEqualTo("|");
        assertThat(map.get(5)).isEqualTo("<unk>");
        assertThat(map.get(6)).isEqualTo("'");
    }

    @Test
    void of_createsFromMap() {
        io.github.inference4j.preprocessing.audio.Vocabulary vocab = io.github.inference4j.preprocessing.audio.Vocabulary.of(Map.of(
                0, "<pad>",
                1, "a",
                2, "b"
        ));

        assertThat(vocab.size()).isEqualTo(3);
        assertThat(vocab.get(0)).isEqualTo("<pad>");
        assertThat(vocab.get(1)).isEqualTo("a");
        assertThat(vocab.get(2)).isEqualTo("b");
    }

    @Test
    void get_throwsForUnknownIndex() {
        io.github.inference4j.preprocessing.audio.Vocabulary vocab = io.github.inference4j.preprocessing.audio.Vocabulary.of(Map.of(0, "a"));

        assertThatThrownBy(() -> vocab.get(99)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void parseVocabJson_emptyObject() {
        Map<Integer, String> map = io.github.inference4j.preprocessing.audio.Vocabulary.parseVocabJson("{}");
        assertThat(map).isEmpty();
    }

    @Test
    void parseVocabJson_withWhitespace() {
        String json = """
                {
                  "a" : 1 ,
                  "b" : 2
                }
                """;

        Map<Integer, String> map = io.github.inference4j.preprocessing.audio.Vocabulary.parseVocabJson(json);

        assertThat(map.size()).isEqualTo(2);
        assertThat(map.get(1)).isEqualTo("a");
        assertThat(map.get(2)).isEqualTo("b");
    }
}
