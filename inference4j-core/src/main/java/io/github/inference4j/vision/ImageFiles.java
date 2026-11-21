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

import io.github.inference4j.exception.InferenceException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Reads image files for the vision wrappers, so every detector loads files the same way.
 */
final class ImageFiles {

    private ImageFiles() {
    }

    /**
     * Reads an image file with {@link ImageIO}.
     *
     * @throws InferenceException if the file cannot be read or its format is not supported
     */
    static BufferedImage read(Path path) {
        try {
            BufferedImage image = ImageIO.read(path.toFile());
            if (image == null) {
                throw new InferenceException("Unsupported image format: " + path);
            }
            return image;
        } catch (IOException e) {
            throw new InferenceException("Failed to read image: " + path, e);
        }
    }
}
