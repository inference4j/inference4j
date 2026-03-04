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

import io.github.inference4j.InferenceTask;

import java.util.List;

/**
 * Extracts named entities from text using token-level classification.
 *
 * <p>Named Entity Recognition (NER) identifies spans of text that refer to
 * real-world entities such as people, organizations, locations, and
 * miscellaneous entities, returning each as a {@link NamedEntity} with
 * its label, character offsets, and confidence score.
 *
 * @see NamedEntity
 * @see BertNerRecognizer
 */
public interface NamedEntityRecognizer extends InferenceTask<String, List<NamedEntity>> {

    /**
     * Extracts named entities from the given text.
     *
     * @param text the input text
     * @return list of named entities found in the text
     */
    List<NamedEntity> recognize(String text);

    @Override
    default List<NamedEntity> run(String input) {
        return recognize(input);
    }

    @Override
    void close();
}
