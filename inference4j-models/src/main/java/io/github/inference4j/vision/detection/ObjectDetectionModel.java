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

package io.github.inference4j.vision.detection;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

/**
 * Detects objects in images and returns labeled bounding boxes.
 */
public interface ObjectDetectionModel extends AutoCloseable {

    List<Detection> detect(BufferedImage image);

    List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold);

    List<Detection> detect(Path imagePath);

    List<Detection> detect(Path imagePath, float confidenceThreshold, float iouThreshold);

    @Override
    void close();
}
