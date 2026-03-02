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

/**
 * A named entity extracted from text.
 *
 * @param text  the span text (e.g., "London")
 * @param label the entity type (e.g., "LOC", "PER", "ORG", "MISC")
 * @param start character offset start in the original string
 * @param end   character offset end in the original string (exclusive)
 * @param score mean confidence of the constituent tokens
 */
public record NamedEntity(
        String text,
        String label,
        int start,
        int end,
        float score
) {
}
