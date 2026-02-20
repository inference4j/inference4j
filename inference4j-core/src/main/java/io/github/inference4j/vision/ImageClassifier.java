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

package io.github.inference4j.vision;

import io.github.inference4j.Classifier;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

/**
 * Classifies images into labeled categories.
 */
public interface ImageClassifier extends Classifier<BufferedImage, Classification> {

    @Override
    List<Classification> classify(BufferedImage image);

    List<Classification> classify(BufferedImage image, int topK);

    List<Classification> classify(Path imagePath);

    List<Classification> classify(Path imagePath, int topK);

    @Override
    void close();
}
