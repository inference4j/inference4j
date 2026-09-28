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

package io.github.inference4j.modeljars.nlp;

import io.github.inference4j.exception.ModelLoadException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ModelJarsTextGeneratorTest {

    @Test
    void build_withoutModel_throws() {
        assertThatThrownBy(() -> ModelJarsTextGenerator.builder().build())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("model is required");
    }

    @Test
    void build_withMarkerNotOnClasspath_throwsModelLoadException() {
        assertThatThrownBy(() -> ModelJarsTextGenerator.builder()
                .model("org.modeljars.huggingface:does-not.exist:1.0.0")
                .build())
                .isInstanceOf(ModelLoadException.class)
                .hasMessageContaining("does-not.exist");
    }
}
