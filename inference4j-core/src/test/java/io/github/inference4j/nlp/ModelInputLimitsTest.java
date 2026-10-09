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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

class ModelInputLimitsTest {

    @TempDir
    Path dir;

    private Path config(String json) throws IOException {
        return Files.writeString(dir.resolve("config.json"), json);
    }

    @Test
    void readsMaxPositionEmbeddings() throws IOException {
        // BART, Marian, Qwen, Llama
        assertThat(ModelInputLimits.maxPositions(config("{\"model_type\":\"bart\",\"max_position_embeddings\":1024}")))
                .hasValue(1024);
    }

    @Test
    void readsNPositions() throws IOException {
        // GPT-2, and T5 configs that declare it
        assertThat(ModelInputLimits.maxPositions(config("{\"model_type\":\"gpt2\",\"n_positions\":1024}")))
                .hasValue(1024);
    }

    @Test
    void t5WithoutPositionFieldUsesTrainedLength() throws IOException {
        assertThat(ModelInputLimits.maxPositions(config("{\"model_type\":\"t5\",\"relative_attention_num_buckets\":32}")))
                .hasValue(512);
    }

    @Test
    void unknownLimitIsEmpty() throws IOException {
        assertThat(ModelInputLimits.maxPositions(config("{\"model_type\":\"mystery\"}"))).isEmpty();
    }
}
