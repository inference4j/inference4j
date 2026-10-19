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

import java.util.Map;

import io.github.inference4j.Tensor;

/**
 * An image already run through a {@link PromptableSegmenter}'s encoder.
 *
 * <p>Encoding is the expensive step; prompting is cheap. Encode an image once with
 * {@link PromptableSegmenter#encode(java.awt.image.BufferedImage)}, keep this handle, and pass
 * it to {@link PromptableSegmenter#segment(ImageFeatures, Prompt)} for every click or box.
 *
 * <p>The contents are model-specific and opaque: use the handle only with the segmenter that
 * produced it. It holds the encoder's feature maps in Java memory (about 17&nbsp;MB for SAM 2.1),
 * so drop the reference once you have finished prompting that image.
 *
 * @see PromptableSegmenter
 */
public final class ImageFeatures {

    private final Map<String, Tensor> tensors;
    private final int width;
    private final int height;

    ImageFeatures(Map<String, Tensor> tensors, int width, int height) {
        this.tensors = Map.copyOf(tensors);
        this.width = width;
        this.height = height;
    }

    /** Width of the encoded image, in pixels. */
    public int width() {
        return width;
    }

    /** Height of the encoded image, in pixels. */
    public int height() {
        return height;
    }

    Map<String, Tensor> tensors() {
        return tensors;
    }
}
