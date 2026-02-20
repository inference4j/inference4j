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

import static org.junit.jupiter.api.Assertions.*;

class VocabularyTest {

    @Test
    void parseVocabJson_basicVocabulary() {
        String json = """
                {"a": 1, "b": 2, "c": 3}
                """;

        Map<Integer, String> map = io.github.inference4j.preprocessing.audio.Vocabulary.parseVocabJson(json);

        assertEquals(3, map.size());
        assertEquals("a", map.get(1));
        assertEquals("b", map.get(2));
        assertEquals("c", map.get(3));
    }

    @Test
    void parseVocabJson_specialCharacters() {
        String json = """
                {"<pad>": 0, "|": 4, "<unk>": 5, "'": 6}
                """;

        Map<Integer, String> map = io.github.inference4j.preprocessing.audio.Vocabulary.parseVocabJson(json);

        assertEquals("<pad>", map.get(0));
        assertEquals("|", map.get(4));
        assertEquals("<unk>", map.get(5));
        assertEquals("'", map.get(6));
    }

    @Test
    void of_createsFromMap() {
        io.github.inference4j.preprocessing.audio.Vocabulary vocab = io.github.inference4j.preprocessing.audio.Vocabulary.of(Map.of(
                0, "<pad>",
                1, "a",
                2, "b"
        ));

        assertEquals(3, vocab.size());
        assertEquals("<pad>", vocab.get(0));
        assertEquals("a", vocab.get(1));
        assertEquals("b", vocab.get(2));
    }

    @Test
    void get_throwsForUnknownIndex() {
        io.github.inference4j.preprocessing.audio.Vocabulary vocab = io.github.inference4j.preprocessing.audio.Vocabulary.of(Map.of(0, "a"));

        assertThrows(IllegalArgumentException.class, () -> vocab.get(99));
    }

    @Test
    void parseVocabJson_emptyObject() {
        Map<Integer, String> map = io.github.inference4j.preprocessing.audio.Vocabulary.parseVocabJson("{}");
        assertTrue(map.isEmpty());
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

        assertEquals(2, map.size());
        assertEquals("a", map.get(1));
        assertEquals("b", map.get(2));
    }
}
