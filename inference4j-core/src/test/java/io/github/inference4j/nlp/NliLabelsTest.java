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

import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.preprocessing.text.ModelConfig;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class NliLabelsTest {

    @Test
    void contradictionIsFoundByNameWhenEntailmentComesFirst() {
        // Hugging Face's positional rule would pick neutral (the last class) here
        NliLabels labels = labels(Map.of(0, "entailment", 1, "contradiction", 2, "neutral"));

        assertThat(labels).isEqualTo(new NliLabels(0, 1));
    }

    @Test
    void contradictionIsFoundByNameWhenItIsLast() {
        NliLabels labels = labels(Map.of(0, "entailment", 1, "neutral", 2, "contradiction"));

        assertThat(labels).isEqualTo(new NliLabels(0, 2));
    }

    @Test
    void mnliOrderFindsBothClasses() {
        NliLabels labels = labels(Map.of(0, "CONTRADICTION", 1, "NEUTRAL", 2, "ENTAILMENT"));

        assertThat(labels).isEqualTo(new NliLabels(2, 0));
    }

    @Test
    void twoClassModelsWithoutContradictionUseTheOtherClass() {
        assertThat(labels(Map.of(0, "entailment", 1, "not_entailment"))).isEqualTo(new NliLabels(0, 1));
        assertThat(labels(Map.of(0, "not_entailment", 1, "entailment"))).isEqualTo(new NliLabels(1, 0));
    }

    @Test
    void modelsWithoutEntailmentAreRejected() {
        assertThatThrownBy(() -> labels(Map.of(0, "positive", 1, "negative")))
                .isInstanceOf(ModelLoadException.class)
                .hasMessageContaining("entailment");
    }

    private static NliLabels labels(Map<Integer, String> id2label) {
        return NliLabels.from(ModelConfig.of(id2label, null));
    }
}
