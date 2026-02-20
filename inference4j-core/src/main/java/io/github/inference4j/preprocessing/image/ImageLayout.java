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

package io.github.inference4j.preprocessing.image;

/**
 * Tensor layout for image data.
 *
 * <ul>
 *   <li>{@link #NCHW} — [batch, channels, height, width] (PyTorch-origin models)</li>
 *   <li>{@link #NHWC} — [batch, height, width, channels] (TensorFlow-origin models)</li>
 * </ul>
 */
public enum ImageLayout {
    NCHW,
    NHWC;

    /**
     * Detects image layout from a 4D tensor shape by locating the channels dimension (1 or 3).
     *
     * @param shape 4D tensor shape [batch, ?, ?, ?]
     * @return detected layout
     * @throws IllegalArgumentException if shape is not 4D
     */
    public static ImageLayout detect(long[] shape) {
        if (shape.length != 4) {
            throw new IllegalArgumentException(
                    "Expected 4D tensor shape, got " + shape.length + "D");
        }
        if (shape[1] == 3 || shape[1] == 1) return NCHW;
        if (shape[3] == 3 || shape[3] == 1) return NHWC;
        // Ambiguous (e.g. [1,3,3,3]) — default to NCHW (PyTorch convention)
        return NCHW;
    }

    /**
     * Extracts the spatial image size (height) from the shape.
     */
    public int imageSize(long[] shape) {
        return (int) (this == NCHW ? shape[2] : shape[1]);
    }
}
