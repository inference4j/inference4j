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

package io.github.inference4j.preprocessing.text;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.inference4j.exception.ModelSourceException;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Reads model configuration from a HuggingFace {@code config.json} file.
 *
 * <p>Extracts the {@code id2label} mapping and {@code problem_type} field used
 * by sequence classification models to determine label names and activation function.
 *
 * <p>Example config.json fragment:
 * <pre>{@code
 * {
 *   "id2label": {"0": "NEGATIVE", "1": "POSITIVE"},
 *   "problem_type": "single_label_classification"
 * }
 * }</pre>
 */
public final class ModelConfig {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Map<Integer, String> id2label;
    private final String problemType;

    private ModelConfig(Map<Integer, String> id2label, String problemType) {
        this.id2label = Map.copyOf(id2label);
        this.problemType = problemType;
    }

    /**
     * Loads configuration from a {@code config.json} file.
     *
     * @param path path to the config.json file
     * @return the parsed configuration
     * @throws ModelSourceException if the file cannot be read or parsed
     */
    public static ModelConfig fromFile(Path path) {
        try {
            JsonNode root = MAPPER.readTree(path.toFile());
            return fromJsonNode(root);
        } catch (IOException e) {
            throw new ModelSourceException("Failed to read config file: " + path, e);
        }
    }

    /**
     * Creates a configuration from pre-built values.
     *
     * @param id2label    mapping from class index to label name
     * @param problemType the problem type (e.g., "single_label_classification", "multi_label_classification")
     * @return the configuration
     */
    public static ModelConfig of(Map<Integer, String> id2label, String problemType) {
        return new ModelConfig(id2label, problemType);
    }

    /**
     * Returns the label for the given class index.
     *
     * @param index class index
     * @return label string
     * @throws IllegalArgumentException if the index is not in the label map
     */
    public String label(int index) {
        String label = id2label.get(index);
        if (label == null) {
            throw new IllegalArgumentException("Class index not in id2label: " + index);
        }
        return label;
    }

    /**
     * Returns the number of classes.
     */
    public int numLabels() {
        return id2label.size();
    }

    /**
     * Returns the problem type string, or {@code null} if not specified.
     */
    public String problemType() {
        return problemType;
    }

    /**
     * Returns whether this is a multi-label classification problem (sigmoid activation).
     * Falls back to single-label (softmax) if problem_type is absent.
     */
    public boolean isMultiLabel() {
        return "multi_label_classification".equals(problemType);
    }

    /**
     * Parses a config.json string. Package-visible for testing.
     */
    static ModelConfig parse(String json) {
        try {
            JsonNode root = MAPPER.readTree(json);
            return fromJsonNode(root);
        } catch (IOException e) {
            throw new ModelSourceException("Failed to parse config JSON", e);
        }
    }

    private static ModelConfig fromJsonNode(JsonNode root) {
        Map<Integer, String> id2label = new LinkedHashMap<>();
        JsonNode id2labelNode = root.get("id2label");
        if (id2labelNode != null && id2labelNode.isObject()) {
            Iterator<Map.Entry<String, JsonNode>> fields = id2labelNode.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> entry = fields.next();
                id2label.put(Integer.parseInt(entry.getKey()), entry.getValue().asText());
            }
        }

        String problemType = null;
        JsonNode problemTypeNode = root.get("problem_type");
        if (problemTypeNode != null && problemTypeNode.isTextual()) {
            problemType = problemTypeNode.asText();
        }

        return new ModelConfig(id2label, problemType);
    }
}
