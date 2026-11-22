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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.inference4j.exception.ModelLoadException;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.OptionalInt;

/**
 * Reads a model's maximum sequence length (its number of position embeddings) from
 * {@code config.json}, so generators truncate at the model's real limit instead of a tokenizer
 * default.
 */
final class ModelInputLimits {

    /** T5 uses relative positions, so it has no hard limit; 512 is the length it was trained on. */
    static final int T5_TRAINED_LENGTH = 512;

    private ModelInputLimits() {
    }

    /**
     * Returns the number of decoder positions of an encoder-decoder model: {@code max_position_embeddings}
     * or {@code n_positions}, a hard limit for learned or sinusoidal positions (BART, MarianMT). Empty for
     * relative positions (T5, which declares {@code relative_attention_num_buckets}), which have no hard limit,
     * and when the config declares none.
     */
    static OptionalInt decoderPositions(Path configPath) {
        return decoderPositions(read(configPath));
    }

    /** Like {@link #decoderPositions(Path)}, for a parsed {@code config.json}. */
    static OptionalInt decoderPositions(JsonNode root) {
        if (root.has("relative_attention_num_buckets")) {
            return OptionalInt.empty();
        }
        return positionField(root);
    }

    /**
     * Returns {@code max_position_embeddings} (BART, Marian, Qwen, Llama…) or {@code n_positions}
     * (GPT-2, T5), falling back to {@link #T5_TRAINED_LENGTH} for T5 models that declare neither.
     * Empty when the limit is unknown.
     */
    static OptionalInt maxPositions(Path configPath) {
        return maxPositions(read(configPath));
    }

    /** Like {@link #maxPositions(Path)}, for a parsed {@code config.json}. */
    static OptionalInt maxPositions(JsonNode root) {
        OptionalInt positions = positionField(root);
        if (positions.isPresent()) {
            return positions;
        }
        JsonNode modelType = root.get("model_type");
        if (modelType != null && modelType.asText().startsWith("t5")) {
            return OptionalInt.of(T5_TRAINED_LENGTH);
        }
        return OptionalInt.empty();
    }

    private static OptionalInt positionField(JsonNode root) {
        for (String field : new String[]{"max_position_embeddings", "n_positions"}) {
            JsonNode node = root.get(field);
            if (node != null && node.canConvertToInt() && node.intValue() > 0) {
                return OptionalInt.of(node.intValue());
            }
        }
        return OptionalInt.empty();
    }

    private static JsonNode read(Path configPath) {
        try (InputStream is = Files.newInputStream(configPath)) {
            return new ObjectMapper().readTree(is);
        } catch (IOException e) {
            throw new ModelLoadException("Failed to read config.json: " + e.getMessage(), e);
        }
    }
}
