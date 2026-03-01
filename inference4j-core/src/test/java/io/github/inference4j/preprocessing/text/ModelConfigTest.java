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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ModelConfigTest {

    @Test
    void parse_sentimentConfig() {
        String json = """
                {
                  "id2label": {
                    "0": "NEGATIVE",
                    "1": "POSITIVE"
                  },
                  "problem_type": "single_label_classification"
                }
                """;

        io.github.inference4j.preprocessing.text.ModelConfig config = io.github.inference4j.preprocessing.text.ModelConfig.parse(json);

        assertThat(config.numLabels()).isEqualTo(2);
        assertThat(config.label(0)).isEqualTo("NEGATIVE");
        assertThat(config.label(1)).isEqualTo("POSITIVE");
        assertThat(config.problemType()).isEqualTo("single_label_classification");
        assertThat(config.isMultiLabel()).isFalse();
    }

    @Test
    void parse_multiLabelConfig() {
        String json = """
                {
                  "id2label": {
                    "0": "admiration",
                    "1": "amusement",
                    "2": "anger"
                  },
                  "problem_type": "multi_label_classification"
                }
                """;

        io.github.inference4j.preprocessing.text.ModelConfig config = io.github.inference4j.preprocessing.text.ModelConfig.parse(json);

        assertThat(config.numLabels()).isEqualTo(3);
        assertThat(config.label(0)).isEqualTo("admiration");
        assertThat(config.isMultiLabel()).isTrue();
    }

    @Test
    void parse_missingProblemType_defaultsToNull() {
        String json = """
                {
                  "id2label": {
                    "0": "NEGATIVE",
                    "1": "POSITIVE"
                  }
                }
                """;

        io.github.inference4j.preprocessing.text.ModelConfig config = io.github.inference4j.preprocessing.text.ModelConfig.parse(json);

        assertThat(config.problemType()).isNull();
        assertThat(config.isMultiLabel()).isFalse();
    }

    @Test
    void parse_missingId2Label_emptyLabels() {
        String json = """
                {
                  "model_type": "distilbert"
                }
                """;

        io.github.inference4j.preprocessing.text.ModelConfig config = io.github.inference4j.preprocessing.text.ModelConfig.parse(json);

        assertThat(config.numLabels()).isEqualTo(0);
    }

    @Test
    void parse_configWithOtherFields() {
        String json = """
                {
                  "model_type": "distilbert",
                  "architectures": ["DistilBertForSequenceClassification"],
                  "id2label": {
                    "0": "NEGATIVE",
                    "1": "POSITIVE"
                  },
                  "label2id": {
                    "NEGATIVE": 0,
                    "POSITIVE": 1
                  },
                  "problem_type": "single_label_classification",
                  "num_labels": 2
                }
                """;

        io.github.inference4j.preprocessing.text.ModelConfig config = io.github.inference4j.preprocessing.text.ModelConfig.parse(json);

        assertThat(config.numLabels()).isEqualTo(2);
        assertThat(config.label(0)).isEqualTo("NEGATIVE");
        assertThat(config.label(1)).isEqualTo("POSITIVE");
    }

    @Test
    void label_unknownIndex_throws() {
        io.github.inference4j.preprocessing.text.ModelConfig config = io.github.inference4j.preprocessing.text.ModelConfig.parse("""
                {"id2label": {"0": "A", "1": "B"}}
                """);

        assertThatThrownBy(() -> config.label(5)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void of_createsFromMap() {
        io.github.inference4j.preprocessing.text.ModelConfig config = io.github.inference4j.preprocessing.text.ModelConfig.of(
                java.util.Map.of(0, "NEG", 1, "POS"),
                "single_label_classification"
        );

        assertThat(config.numLabels()).isEqualTo(2);
        assertThat(config.label(0)).isEqualTo("NEG");
        assertThat(config.isMultiLabel()).isFalse();
    }
}
