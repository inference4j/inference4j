package io.github.inference4j.audio;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class VocabularyTest {

    @Test
    void parseVocabJson_basicVocabulary() {
        String json = """
                {"a": 1, "b": 2, "c": 3}
                """;

        Map<Integer, String> map = Vocabulary.parseVocabJson(json);

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

        Map<Integer, String> map = Vocabulary.parseVocabJson(json);

        assertEquals("<pad>", map.get(0));
        assertEquals("|", map.get(4));
        assertEquals("<unk>", map.get(5));
        assertEquals("'", map.get(6));
    }

    @Test
    void of_createsFromMap() {
        Vocabulary vocab = Vocabulary.of(Map.of(
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
        Vocabulary vocab = Vocabulary.of(Map.of(0, "a"));

        assertThrows(IllegalArgumentException.class, () -> vocab.get(99));
    }

    @Test
    void parseVocabJson_emptyObject() {
        Map<Integer, String> map = Vocabulary.parseVocabJson("{}");
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

        Map<Integer, String> map = Vocabulary.parseVocabJson(json);

        assertEquals(2, map.size());
        assertEquals("a", map.get(1));
        assertEquals("b", map.get(2));
    }
}
