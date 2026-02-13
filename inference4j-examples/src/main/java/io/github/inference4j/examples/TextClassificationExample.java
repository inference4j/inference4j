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

import io.github.inference4j.text.DistilBertClassifier;
import io.github.inference4j.text.TextClassification;

import java.util.List;

/**
 * Demonstrates sentiment analysis with DistilBERT fine-tuned on SST-2.
 *
 * Requires distilbert-base-uncased-finetuned-sst-2-english ONNX model (~268 MB).
 * See inference4j-examples/README.md for download instructions.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.TextClassificationExample
 */
public class TextClassificationExample {

    public static void main(String[] args) {
        String modelDir = "inference4j-examples/models/distilbert-sst2";

        String[] texts = {
                "This movie was absolutely fantastic! I loved every minute of it.",
                "Terrible experience. The food was cold and the service was awful.",
                "The weather is nice today.",
                "I'm so disappointed with this product, it broke after one day.",
                "Best purchase I've ever made, highly recommend!",
                "The plot was predictable but the acting was superb.",
        };

        try (DistilBertClassifier model = DistilBertClassifier.fromPretrained(modelDir)) {
            System.out.println("DistilBERT-SST2 loaded successfully.");
            System.out.println();

            for (String text : texts) {
                List<TextClassification> results = model.classify(text);

                TextClassification top = results.get(0);
                System.out.printf("  %-7s (%.4f)  \"%s\"%n",
                        top.label(), top.confidence(), text);
            }
        }
    }
}
