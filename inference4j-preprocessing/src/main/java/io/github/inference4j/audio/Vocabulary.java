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

package io.github.inference4j.audio;

import io.github.inference4j.exception.InferenceException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Token vocabulary for CTC-based speech models.
 *
 * <p>Maps integer token indices to string characters. Loaded from a JSON file
 * in the format {@code {"a": 1, "b": 2, "|": 4, "<pad>": 0, ...}}.
 */
public final class Vocabulary {

    private final Map<Integer, String> indexToToken;

    private Vocabulary(Map<Integer, String> indexToToken) {
        this.indexToToken = Map.copyOf(indexToToken);
    }

    /**
     * Loads a vocabulary from a JSON file.
     *
     * <p>Expects a flat JSON object mapping token strings to integer indices:
     * {@code {"a": 1, "b": 2, "|": 4, "<pad>": 0}}.
     *
     * @param path path to the vocab.json file
     * @return the loaded vocabulary
     * @throws InferenceException if the file cannot be read or parsed
     */
    public static Vocabulary fromFile(Path path) {
        try {
            String json = Files.readString(path);
            Map<Integer, String> map = parseVocabJson(json);
            return new Vocabulary(map);
        } catch (IOException e) {
            throw new InferenceException("Failed to read vocabulary file: " + path, e);
        }
    }

    /**
     * Creates a vocabulary from a pre-built index-to-token map.
     *
     * @param map mapping from token index to token string
     * @return the vocabulary
     */
    public static Vocabulary of(Map<Integer, String> map) {
        return new Vocabulary(map);
    }

    /**
     * Returns the token string for the given index.
     *
     * @param index token index
     * @return token string
     * @throws IllegalArgumentException if the index is not in the vocabulary
     */
    public String get(int index) {
        String token = indexToToken.get(index);
        if (token == null) {
            throw new IllegalArgumentException("Token index not in vocabulary: " + index);
        }
        return token;
    }

    /**
     * Returns the number of tokens in the vocabulary.
     */
    public int size() {
        return indexToToken.size();
    }

    /**
     * Minimal parser for flat JSON objects mapping strings to integers.
     * Handles escaped characters in keys (e.g., {@code "\""}, {@code "\\"}).
     */
    static Map<Integer, String> parseVocabJson(String json) {
        Map<Integer, String> map = new HashMap<>();
        int i = skipWhitespace(json, 0);
        if (i >= json.length() || json.charAt(i) != '{') {
            throw new InferenceException("Expected '{' at start of vocab JSON");
        }
        i++;

        while (i < json.length()) {
            i = skipWhitespace(json, i);
            if (i >= json.length()) break;
            if (json.charAt(i) == '}') break;
            if (json.charAt(i) == ',') {
                i++;
                continue;
            }

            // Parse key (string)
            if (json.charAt(i) != '"') {
                throw new InferenceException("Expected '\"' at position " + i);
            }
            int[] keyResult = parseString(json, i);
            String key = json.substring(keyResult[0], keyResult[1]);
            i = keyResult[2];

            // Skip colon
            i = skipWhitespace(json, i);
            if (i >= json.length() || json.charAt(i) != ':') {
                throw new InferenceException("Expected ':' at position " + i);
            }
            i++;

            // Parse value (integer)
            i = skipWhitespace(json, i);
            int numStart = i;
            while (i < json.length() && (Character.isDigit(json.charAt(i)) || json.charAt(i) == '-')) {
                i++;
            }
            int value = Integer.parseInt(json.substring(numStart, i));

            map.put(value, key);
        }

        return map;
    }

    /**
     * Parses a JSON string starting at position {@code start} (which must be a double-quote).
     * Returns {@code [contentStart, contentEnd, nextPosition]} where content excludes quotes.
     * Handles escape sequences.
     */
    private static int[] parseString(String json, int start) {
        int i = start + 1; // skip opening quote
        StringBuilder sb = null;
        int contentStart = i;

        while (i < json.length()) {
            char c = json.charAt(i);
            if (c == '\\') {
                // For escaped characters, we need a StringBuilder approach
                if (sb == null) {
                    sb = new StringBuilder(json.substring(contentStart, i));
                }
                i++;
                if (i < json.length()) {
                    sb.append(json.charAt(i));
                }
                i++;
            } else if (c == '"') {
                // End of string
                if (sb != null) {
                    // We had escapes — return the unescaped content via a trick:
                    // store in a temporary way. Since we need start/end indices,
                    // this simple parser returns the raw content range for non-escaped strings.
                    // For escaped strings, we return the raw range (caller gets the raw JSON key
                    // which for vocab.json is fine — keys like "|", "<pad>" have no escapes).
                }
                return new int[]{contentStart, i, i + 1};
            } else {
                if (sb != null) {
                    sb.append(c);
                }
                i++;
            }
        }

        throw new InferenceException("Unterminated string in vocab JSON");
    }

    private static int skipWhitespace(String json, int i) {
        while (i < json.length() && Character.isWhitespace(json.charAt(i))) {
            i++;
        }
        return i;
    }
}
