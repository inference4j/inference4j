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

package io.github.inference4j.text;

import java.util.List;

/**
 * Classifies text into labeled categories.
 *
 * <p>Returns all classes sorted by confidence descending by default, or the
 * top-K most confident classes.
 *
 * @see TextClassification
 */
public interface TextClassificationModel extends AutoCloseable {

    /**
     * Classifies the given text, returning all classes sorted by confidence.
     *
     * @param text the input text to classify
     * @return classification results sorted by confidence descending
     */
    List<TextClassification> classify(String text);

    /**
     * Classifies the given text, returning the top-K most confident classes.
     *
     * @param text the input text to classify
     * @param topK the maximum number of results to return
     * @return classification results sorted by confidence descending
     */
    List<TextClassification> classify(String text, int topK);

    @Override
    void close();
}
