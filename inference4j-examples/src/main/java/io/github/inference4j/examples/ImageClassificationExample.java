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

import io.github.inference4j.vision.classification.Classification;
import io.github.inference4j.vision.classification.EfficientNet;
import io.github.inference4j.vision.classification.ResNet;

import java.nio.file.Path;
import java.util.List;

/**
 * Demonstrates image classification with ResNet-50 and EfficientNet-B0.
 *
 * Requires ONNX models and a sample image — see inference4j-examples/README.md for setup.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.ImageClassificationExample
 */
public class ImageClassificationExample {

    public static void main(String[] args) {
        String imagePath = "inference4j-examples/images/sample.jpg";

        // --- ResNet-50 ---
        System.out.println("=== ResNet-50 ===");
        try (ResNet resnet = ResNet.fromPretrained("inference4j-examples/models/resnet50")) {
            System.out.println("ResNet-50 loaded successfully.");
            System.out.println();

            List<Classification> results = resnet.classify(Path.of(imagePath), 5);

            System.out.println("Top-5 predictions:");
            printResults(results);
        }

        System.out.println();

        // --- EfficientNet-Lite4 ---
        System.out.println("=== EfficientNet-Lite4 ===");
        try (EfficientNet efficientnet = EfficientNet.fromPretrained("inference4j-examples/models/efficientnet-lite4")) {
            System.out.println("EfficientNet-Lite4 loaded successfully.");
            System.out.println();

            List<Classification> results = efficientnet.classify(Path.of(imagePath), 5);

            System.out.println("Top-5 predictions:");
            printResults(results);
        }
    }

    private static void printResults(List<Classification> results) {
        for (int i = 0; i < results.size(); i++) {
            Classification c = results.get(i);
            System.out.printf("  %d. %s (index=%d, confidence=%.4f)%n",
                    i + 1, c.label(), c.index(), c.confidence());
        }
    }
}
