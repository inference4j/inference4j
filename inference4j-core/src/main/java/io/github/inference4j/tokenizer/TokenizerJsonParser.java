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

package io.github.inference4j.tokenizer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.inference4j.exception.ModelSourceException;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parses HuggingFace {@code tokenizer.json} files into a
 * {@link SentencePieceBpeTokenizer.Builder}.
 *
 * <p>The HuggingFace tokenizer JSON format is a universal serialization of
 * tokenizer configurations used across model families. This parser extracts
 * the vocabulary, merge rules, and special tokens needed to construct a
 * SentencePiece BPE tokenizer.
 *
 * <p>Supports both merge formats: space-separated strings ({@code "▁ t"},
 * tokenizers &lt; 0.20) and two-element arrays ({@code ["▁", "t"]},
 * tokenizers &ge; 0.20). The array format was introduced in
 * <a href="https://github.com/huggingface/tokenizers/pull/909">tokenizers#909</a>
 * to handle merge tokens containing spaces.
 */
public final class TokenizerJsonParser {

    private TokenizerJsonParser() {
    }

    /**
     * Parses a HuggingFace {@code tokenizer.json} and returns a pre-configured builder.
     *
     * <p>Extracts:
     * <ul>
     *   <li>{@code model.vocab} → token-to-ID mapping</li>
     *   <li>{@code model.merges} → BPE merge pairs (string or array format)</li>
     *   <li>{@code added_tokens[].content} where {@code special: true} → added tokens</li>
     * </ul>
     *
     * @param tokenizerJson path to the {@code tokenizer.json} file
     * @return a builder pre-configured with vocab, merges, and added tokens
     * @throws ModelSourceException if the file cannot be read or parsed
     */
    public static SentencePieceBpeTokenizer.Builder parse(Path tokenizerJson) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(tokenizerJson.toFile());

            JsonNode modelNode = root.get("model");
            if (modelNode == null) {
                throw new ModelSourceException(
                        "tokenizer.json missing 'model' section: " + tokenizerJson);
            }

            // Extract vocabulary
            JsonNode vocabNode = modelNode.get("vocab");
            if (vocabNode == null || !vocabNode.isObject()) {
                throw new ModelSourceException(
                        "tokenizer.json missing 'model.vocab': " + tokenizerJson);
            }

            Map<String, Integer> vocab = new LinkedHashMap<>();
            Iterator<Map.Entry<String, JsonNode>> fields = vocabNode.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> entry = fields.next();
                vocab.put(entry.getKey(), entry.getValue().intValue());
            }

            // Extract merges — handles both string ("▁ t") and array (["▁", "t"]) formats
            JsonNode mergesNode = modelNode.get("merges");
            List<String> merges = new ArrayList<>();
            if (mergesNode != null && mergesNode.isArray()) {
                for (JsonNode merge : mergesNode) {
                    if (merge.isTextual()) {
                        merges.add(merge.textValue());
                    } else if (merge.isArray() && merge.size() == 2) {
                        merges.add(merge.get(0).textValue() + " " + merge.get(1).textValue());
                    }
                }
            }

            // Extract special added tokens
            SentencePieceBpeTokenizer.Builder builder = SentencePieceBpeTokenizer.builder()
                    .vocab(vocab)
                    .merges(merges);

            JsonNode addedTokensNode = root.get("added_tokens");
            if (addedTokensNode != null && addedTokensNode.isArray()) {
                for (JsonNode tokenNode : addedTokensNode) {
                    JsonNode specialNode = tokenNode.get("special");
                    if (specialNode != null && specialNode.asBoolean()) {
                        JsonNode contentNode = tokenNode.get("content");
                        if (contentNode != null) {
                            builder.addedToken(contentNode.textValue());
                        }
                    }
                }
            }

            return builder;
        } catch (IOException e) {
            throw new ModelSourceException(
                    "Failed to parse tokenizer.json: " + e.getMessage(), e);
        }
    }
}
