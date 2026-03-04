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

package io.github.inference4j.examples;

import io.github.inference4j.nlp.BertNerRecognizer;
import io.github.inference4j.nlp.NamedEntity;

import java.util.List;

/**
 * Demonstrates Named Entity Recognition with DistilBERT-NER.
 *
 * Extracts persons, organizations, locations, and miscellaneous entities from text
 * using IOB2 tagging with a cased WordPiece tokenizer.
 *
 * Requires distilbert-NER ONNX model (~260 MB).
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.NamedEntityRecognitionExample
 */
public class NamedEntityRecognitionExample {

    public static void main(String[] args) {
        String[] texts = {
                "John works at Google in London.",
                "Marie Curie worked at the University of Paris in France.",
                "Tesla announced record earnings at their Austin headquarters.",
                "The United Nations held a conference in Geneva last week.",
                "Leonardo da Vinci studied art in Florence and Milan.",
                "The weather is nice today.",
        };

        try (BertNerRecognizer ner = BertNerRecognizer.builder().build()) {
            System.out.println("DistilBERT-NER loaded successfully.");
            System.out.println();

            for (String text : texts) {
                List<NamedEntity> entities = ner.recognize(text);

                System.out.printf("  \"%s\"%n", text);
                if (entities.isEmpty()) {
                    System.out.println("    (no entities)");
                } else {
                    for (NamedEntity entity : entities) {
                        System.out.printf("    %-25s %-5s (%.2f%%)%n",
                                entity.text(), entity.label(), entity.score() * 100);
                    }
                }
                System.out.println();
            }
        }
    }
}
