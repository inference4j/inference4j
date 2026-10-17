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

package io.github.inference4j.preprocessing.tokenizer;

import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.tokenizer.SpecialTokens;
import io.github.inference4j.tokenizer.TokenizerJsonParser;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class SpecialTokensTest {

    @TempDir
    Path tempDir;

    @Test
    void readsATemplateForSingleTextsAndPairs() throws Exception {
        // DeBERTa-v3's post-processor
        SpecialTokens special = parse("""
                {"type": "TemplateProcessing",
                 "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                            {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
                 "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                          {"SpecialToken": {"id": "[SEP]", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}},
                          {"SpecialToken": {"id": "[SEP]", "type_id": 1}}],
                 "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]},
                                    "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]}}}""");

        assertThat(special.wrap(new long[]{10, 11})).containsExactly(1, 10, 11, 2);
        assertThat(special.wrap(new long[]{10, 11}, new long[]{20})).containsExactly(1, 10, 11, 2, 20, 2);
        assertThat(special.singleCount()).isEqualTo(2);
        assertThat(special.pairCount()).isEqualTo(3);
    }

    @Test
    void readsRobertaProcessing() throws Exception {
        SpecialTokens special = parse("""
                {"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0]}""");

        assertThat(special.wrap(new long[]{7}, new long[]{8})).containsExactly(0, 7, 2, 2, 8, 2);
    }

    @Test
    void readsBertProcessing() throws Exception {
        SpecialTokens special = parse("""
                {"type": "BertProcessing", "sep": ["[SEP]", 102], "cls": ["[CLS]", 101]}""");

        assertThat(special.wrap(new long[]{7})).containsExactly(101, 7, 102);
        assertThat(special.wrap(new long[]{7}, new long[]{8})).containsExactly(101, 7, 102, 8, 102);
    }

    @Test
    void rejectsAnUnknownPostProcessor() throws Exception {
        assertThatThrownBy(() -> parse("""
                {"type": "ByteLevel"}"""))
                .isInstanceOf(ModelSourceException.class);
    }

    private SpecialTokens parse(String postProcessor) throws Exception {
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, "{\"post_processor\": " + postProcessor + "}");
        return TokenizerJsonParser.parseSpecialTokens(file);
    }
}
