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

import io.github.inference4j.InferenceTask;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Encodes images into dense vector embeddings.
 *
 * <p>Image embedders map images to fixed-dimensional float vectors in a learned
 * embedding space. These vectors can be used for similarity search, clustering,
 * or cross-modal retrieval (e.g., matching images with text descriptions).
 *
 * @see io.github.inference4j.nlp.TextEmbedder
 */
public interface ImageEmbedder extends InferenceTask<BufferedImage, float[]> {

    float[] encode(BufferedImage image);

    default float[] encode(Path imagePath) {
        try {
            BufferedImage image = ImageIO.read(imagePath.toFile());
            return encode(image);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read image: " + imagePath, e);
        }
    }

    List<float[]> encodeBatch(List<BufferedImage> images);

    @Override
    default float[] run(BufferedImage input) {
        return encode(input);
    }

    @Override
    void close();
}
