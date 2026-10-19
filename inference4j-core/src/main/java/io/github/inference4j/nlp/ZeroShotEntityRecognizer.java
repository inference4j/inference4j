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

import java.util.List;

/**
 * Extracts named entities of types chosen at call time, such as {@code "person"}, {@code "invoice id"} or
 * {@code "medication"}, without a model trained for those types.
 *
 * <pre>{@code
 * try (ZeroShotEntityRecognizer ner = GlinerEntityRecognizer.builder().build()) {
 *     List<NamedEntity> entities = ner.recognize(text, List.of("person", "organization", "invoice id"));
 * }
 * }</pre>
 *
 * <p>Entities come back in text order. Each {@link NamedEntity#label()} is one of the labels passed in.
 *
 * @see GlinerEntityRecognizer
 * @see NamedEntityRecognizer
 */
public interface ZeroShotEntityRecognizer extends AutoCloseable {

    /**
     * Extracts entities of the given types.
     *
     * @param text   the text to search
     * @param labels the entity types to look for, in plain words
     * @return the entities found, in text order
     */
    List<NamedEntity> recognize(String text, List<String> labels);

    @Override
    void close();
}
