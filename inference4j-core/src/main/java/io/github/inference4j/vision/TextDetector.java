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
 * Detects text regions in images.
 *
 * <p>Unlike {@link ObjectDetector},
 * text detection models output pixel-level heatmaps that are post-processed
 * into text regions. There is no IoU threshold because connected
 * components are naturally non-overlapping. Each {@link TextRegion} carries an axis-aligned box and
 * its outline, which is rotated for detectors that find rotated lines.
 *
 * <p>Detector-specific settings, such as thresholds, belong to each implementation.
 *
 * @see TextRegion
 * @see CraftTextDetector
 * @see PaddleTextDetector
 */
public interface TextDetector extends Detector<BufferedImage, TextRegion> {

    @Override
    List<TextRegion> detect(BufferedImage image);

    List<TextRegion> detect(Path imagePath);

    @Override
    void close();
}
