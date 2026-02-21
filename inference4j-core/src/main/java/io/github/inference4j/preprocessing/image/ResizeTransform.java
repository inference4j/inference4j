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

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

/**
 * Resizes an image to the specified dimensions.
 *
 * <p>Default interpolation is bilinear. Use {@link Interpolation#BICUBIC} for
 * models that expect bicubic resizing (e.g. CLIP).
 */
public class ResizeTransform implements ImageTransform {

    private final int width;
    private final int height;
    private final Interpolation interpolation;

    public ResizeTransform(int width, int height) {
        this(width, height, Interpolation.BILINEAR);
    }

    public ResizeTransform(int width, int height, Interpolation interpolation) {
        this.width = width;
        this.height = height;
        this.interpolation = interpolation;
    }

    @Override
    public BufferedImage apply(BufferedImage image) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, interpolation.renderingHint());
            g.drawImage(image, 0, 0, width, height, null);
        } finally {
            g.dispose();
        }
        return resized;
    }
}
