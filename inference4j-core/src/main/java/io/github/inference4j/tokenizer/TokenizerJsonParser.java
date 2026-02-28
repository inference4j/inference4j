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
 * Parses HuggingFace {@code tokenizer.json} files into tokenizer builders.
 *
 * <p>The HuggingFace tokenizer JSON format is a universal serialization of
 * tokenizer configurations used across model families. This parser extracts
 * the vocabulary, merge rules, scores, and special tokens needed to construct
 * SentencePiece BPE or Unigram tokenizers.
 *
 * <p>For BPE models, supports both merge formats: space-separated strings
 * ({@code "▁ t"}, tokenizers &lt; 0.20) and two-element arrays
 * ({@code ["▁", "t"]}, tokenizers &ge; 0.20). The array format was introduced in
 * <a href="https://github.com/huggingface/tokenizers/pull/909">tokenizers#909</a>
 * to handle merge tokens containing spaces.
 *
 * <p>For Unigram models, extracts per-token log-probability scores needed by
 * {@link UnigramTokenizer} for Viterbi decoding.
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

            // Extract vocabulary — supports both BPE (object) and Unigram (array) formats
            JsonNode vocabNode = modelNode.get("vocab");
            if (vocabNode == null) {
                throw new ModelSourceException(
                        "tokenizer.json missing 'model.vocab': " + tokenizerJson);
            }

            Map<String, Integer> vocab = new LinkedHashMap<>();
            if (vocabNode.isObject()) {
                // BPE format: {"token": id, ...}
                Iterator<Map.Entry<String, JsonNode>> fields = vocabNode.fields();
                while (fields.hasNext()) {
                    Map.Entry<String, JsonNode> entry = fields.next();
                    vocab.put(entry.getKey(), entry.getValue().intValue());
                }
            } else if (vocabNode.isArray()) {
                // Unigram format: [["token", score], ...]
                // The array index IS the token ID
                for (int i = 0; i < vocabNode.size(); i++) {
                    JsonNode entry = vocabNode.get(i);
                    if (entry.isArray() && entry.size() >= 1) {
                        vocab.put(entry.get(0).textValue(), i);
                    }
                }
            } else {
                throw new ModelSourceException(
                        "tokenizer.json 'model.vocab' has unexpected format: " + tokenizerJson);
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

    /**
     * Parses a HuggingFace {@code tokenizer.json} with a Unigram model and returns
     * a pre-configured {@link UnigramTokenizer.Builder}.
     *
     * <p>Extracts:
     * <ul>
     *   <li>{@code model.vocab} → token-to-ID mapping and per-token scores</li>
     *   <li>{@code model.unk_id} → unknown token ID</li>
     *   <li>{@code added_tokens[].content} where {@code special: true} → added tokens</li>
     * </ul>
     *
     * @param tokenizerJson path to the {@code tokenizer.json} file
     * @return a builder pre-configured with vocab, scores, and added tokens
     * @throws ModelSourceException if the file cannot be read or parsed
     */
    public static UnigramTokenizer.Builder parseUnigram(Path tokenizerJson) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(tokenizerJson.toFile());

            JsonNode modelNode = root.get("model");
            if (modelNode == null) {
                throw new ModelSourceException(
                        "tokenizer.json missing 'model' section: " + tokenizerJson);
            }

            JsonNode vocabNode = modelNode.get("vocab");
            if (vocabNode == null || !vocabNode.isArray()) {
                throw new ModelSourceException(
                        "tokenizer.json missing or invalid 'model.vocab' array: " + tokenizerJson);
            }

            Map<String, Integer> vocab = new LinkedHashMap<>();
            float[] scores = new float[vocabNode.size()];
            for (int i = 0; i < vocabNode.size(); i++) {
                JsonNode entry = vocabNode.get(i);
                if (entry.isArray() && entry.size() >= 2) {
                    vocab.put(entry.get(0).textValue(), i);
                    scores[i] = (float) entry.get(1).doubleValue();
                }
            }

            int unkId = 0;
            JsonNode unkIdNode = modelNode.get("unk_id");
            if (unkIdNode != null) {
                unkId = unkIdNode.intValue();
            }

            UnigramTokenizer.Builder builder = UnigramTokenizer.builder()
                    .vocab(vocab)
                    .scores(scores)
                    .unkId(unkId);

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
