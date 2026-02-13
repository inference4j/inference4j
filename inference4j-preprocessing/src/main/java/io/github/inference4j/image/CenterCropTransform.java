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

package io.github.inference4j.image;

import java.awt.image.BufferedImage;

/**
 * Crops the center region of an image to the specified dimensions.
 */
public class CenterCropTransform implements ImageTransform {

    private final int width;
    private final int height;

    public CenterCropTransform(int width, int height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public BufferedImage apply(BufferedImage image) {
        int x = (image.getWidth() - width) / 2;
        int y = (image.getHeight() - height) / 2;
        return image.getSubimage(x, y, width, height);
    }
}
