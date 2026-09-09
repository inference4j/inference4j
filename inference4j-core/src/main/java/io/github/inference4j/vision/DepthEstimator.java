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

import java.awt.image.BufferedImage;
import java.nio.file.Path;

import io.github.inference4j.InferenceTask;

/**
 * Estimates per-pixel depth from a single image.
 *
 * <p>Monocular depth estimation predicts how far each pixel is from the camera
 * using one ordinary photograph — no stereo pair, no depth sensor. The result is a
 * {@link DepthMap} the same size as the input.
 *
 * <p>Unlike {@link ImageClassifier} or {@link ObjectDetector}, this produces a dense
 * prediction rather than a list, so it extends {@link InferenceTask} directly instead
 * of {@code Classifier} or {@code Detector}.
 *
 * @see DepthMap
 * @see DepthAnythingEstimator
 */
public interface DepthEstimator extends InferenceTask<BufferedImage, DepthMap> {

    /**
     * Estimates depth for an image.
     *
     * @param image the input image
     * @return per-pixel depth in the image's own coordinates
     */
    DepthMap estimate(BufferedImage image);

    /**
     * Estimates depth for an image file.
     *
     * @param imagePath path to the image
     * @return per-pixel depth in the image's own coordinates
     */
    DepthMap estimate(Path imagePath);

    @Override
    default DepthMap run(BufferedImage input) {
        return estimate(input);
    }

    @Override
    void close();
}
