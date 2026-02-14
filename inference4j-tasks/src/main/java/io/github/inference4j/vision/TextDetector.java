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

import io.github.inference4j.Detector;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

/**
 * Detects text regions in images and returns axis-aligned bounding boxes.
 *
 * <p>Unlike {@link ObjectDetector},
 * text detection models output pixel-level heatmaps that are post-processed
 * into text region bounding boxes. There is no IoU threshold because connected
 * components are naturally non-overlapping.
 *
 * @see TextRegion
 */
public interface TextDetector extends Detector<BufferedImage, TextRegion> {

    @Override
    List<TextRegion> detect(BufferedImage image);

    List<TextRegion> detect(BufferedImage image, float textThreshold, float lowTextThreshold);

    List<TextRegion> detect(Path imagePath);

    List<TextRegion> detect(Path imagePath, float textThreshold, float lowTextThreshold);

    @Override
    void close();
}
