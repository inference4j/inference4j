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

import io.github.inference4j.Tensor;
import io.github.inference4j.preprocessing.image.CenterCropTransform;
import io.github.inference4j.preprocessing.image.ImageLayout;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

class ImageTransformPipelineTest {

    @Test
    void resize_changesImageDimensions() {
        BufferedImage image = createTestImage(100, 80);
        io.github.inference4j.preprocessing.image.ResizeTransform resize = new io.github.inference4j.preprocessing.image.ResizeTransform(50, 40);

        BufferedImage result = resize.apply(image);

        assertEquals(50, result.getWidth());
        assertEquals(40, result.getHeight());
    }

    @Test
    void centerCrop_extractsCenterRegion() {
        BufferedImage image = createTestImage(100, 100);
        CenterCropTransform crop = new CenterCropTransform(50, 50);

        BufferedImage result = crop.apply(image);

        assertEquals(50, result.getWidth());
        assertEquals(50, result.getHeight());
    }

    @Test
    void transform_producesNCHWShape() {
        io.github.inference4j.preprocessing.image.ImageTransformPipeline pipeline = io.github.inference4j.preprocessing.image.ImageTransformPipeline.builder()
                .resize(32, 32)
                .mean(new float[]{0f, 0f, 0f})
                .std(new float[]{1f, 1f, 1f})
                .build();

        BufferedImage image = createTestImage(64, 64);
        Tensor tensor = pipeline.transform(image);
        long[] shape = tensor.shape();

        assertEquals(4, shape.length);
        assertEquals(1, shape[0]); // batch
        assertEquals(3, shape[1]); // channels
        assertEquals(32, shape[2]); // height
        assertEquals(32, shape[3]); // width
    }

    @Test
    void transform_normalizesPixelValues() {
        // Create a solid red image (RGB = 255, 0, 0)
        BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                image.setRGB(x, y, 0xFF0000);
            }
        }

        // No mean/std adjustment — just scale to [0,1]
        io.github.inference4j.preprocessing.image.ImageTransformPipeline pipeline = io.github.inference4j.preprocessing.image.ImageTransformPipeline.builder()
                .mean(new float[]{0f, 0f, 0f})
                .std(new float[]{1f, 1f, 1f})
                .build();

        Tensor tensor = pipeline.transform(image);
        float[] data = tensor.toFloats();

        // Red channel = 1.0, Green = 0.0, Blue = 0.0
        // NCHW: [R_00, R_01, R_10, R_11, G_00, ..., B_00, ...]
        assertEquals(1.0f, data[0], 1e-5f); // R
        assertEquals(0.0f, data[4], 1e-5f); // G
        assertEquals(0.0f, data[8], 1e-5f); // B
    }

    @Test
    void imagenetPreset_producesCorrectShape() {
        io.github.inference4j.preprocessing.image.ImageTransformPipeline pipeline = io.github.inference4j.preprocessing.image.ImageTransformPipeline.imagenet(224);
        BufferedImage image = createTestImage(300, 400);

        Tensor tensor = pipeline.transform(image);
        long[] shape = tensor.shape();

        assertArrayEquals(new long[]{1, 3, 224, 224}, shape);
    }

    @Test
    void imagenetPreset_appliesNormalization() {
        // White image: all channels = 255 → scaled = 1.0
        // ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        // Normalized R = (1.0 - 0.485) / 0.229 ≈ 2.2489
        BufferedImage image = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 256; y++) {
            for (int x = 0; x < 256; x++) {
                image.setRGB(x, y, 0xFFFFFF);
            }
        }

        io.github.inference4j.preprocessing.image.ImageTransformPipeline pipeline = io.github.inference4j.preprocessing.image.ImageTransformPipeline.imagenet(224);
        Tensor tensor = pipeline.transform(image);
        float[] data = tensor.toFloats();

        float expectedR = (1.0f - 0.485f) / 0.229f;
        assertEquals(expectedR, data[0], 0.01f);
    }

    @Test
    void pipeline_chainsMultipleTransforms() {
        io.github.inference4j.preprocessing.image.ImageTransformPipeline pipeline = io.github.inference4j.preprocessing.image.ImageTransformPipeline.builder()
                .resize(100, 100)
                .centerCrop(50, 50)
                .mean(new float[]{0f, 0f, 0f})
                .std(new float[]{1f, 1f, 1f})
                .build();

        BufferedImage image = createTestImage(200, 200);
        Tensor tensor = pipeline.transform(image);
        long[] shape = tensor.shape();

        assertArrayEquals(new long[]{1, 3, 50, 50}, shape);
    }

    @Test
    void transform_nhwc_producesCorrectShape() {
        io.github.inference4j.preprocessing.image.ImageTransformPipeline pipeline = io.github.inference4j.preprocessing.image.ImageTransformPipeline.builder()
                .resize(32, 32)
                .mean(new float[]{0f, 0f, 0f})
                .std(new float[]{1f, 1f, 1f})
                .layout(ImageLayout.NHWC)
                .build();

        BufferedImage image = createTestImage(64, 64);
        Tensor tensor = pipeline.transform(image);
        long[] shape = tensor.shape();

        assertArrayEquals(new long[]{1, 32, 32, 3}, shape);
    }

    @Test
    void transform_nhwc_normalizesPixelValues() {
        // Solid red image (255, 0, 0)
        BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                image.setRGB(x, y, 0xFF0000);
            }
        }

        io.github.inference4j.preprocessing.image.ImageTransformPipeline pipeline = io.github.inference4j.preprocessing.image.ImageTransformPipeline.builder()
                .mean(new float[]{0f, 0f, 0f})
                .std(new float[]{1f, 1f, 1f})
                .layout(ImageLayout.NHWC)
                .build();

        Tensor tensor = pipeline.transform(image);
        float[] data = tensor.toFloats();

        // NHWC: [R, G, B, R, G, B, ...]
        assertEquals(1.0f, data[0], 1e-5f); // R of pixel (0,0)
        assertEquals(0.0f, data[1], 1e-5f); // G of pixel (0,0)
        assertEquals(0.0f, data[2], 1e-5f); // B of pixel (0,0)
    }

    @Test
    void nhwcPipeline_producesCorrectShape() {
        io.github.inference4j.preprocessing.image.ImageTransformPipeline pipeline = io.github.inference4j.preprocessing.image.ImageTransformPipeline.builder()
                .resize(224, 224)
                .centerCrop(224, 224)
                .mean(new float[]{127f / 255f, 127f / 255f, 127f / 255f})
                .std(new float[]{128f / 255f, 128f / 255f, 128f / 255f})
                .layout(ImageLayout.NHWC)
                .build();
        BufferedImage image = createTestImage(300, 400);

        Tensor tensor = pipeline.transform(image);
        long[] shape = tensor.shape();

        assertArrayEquals(new long[]{1, 224, 224, 3}, shape);
    }

    @Test
    void nhwcPipeline_appliesCorrectNormalization() {
        // White image: all channels = 255 → scaled = 1.0
        // (pixel - 127) / 128 = (255 - 127) / 128 = 1.0
        // In mean/std terms: (1.0 - 127/255) / (128/255)
        BufferedImage image = new BufferedImage(224, 224, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                image.setRGB(x, y, 0xFFFFFF);
            }
        }

        io.github.inference4j.preprocessing.image.ImageTransformPipeline pipeline = io.github.inference4j.preprocessing.image.ImageTransformPipeline.builder()
                .mean(new float[]{127f / 255f, 127f / 255f, 127f / 255f})
                .std(new float[]{128f / 255f, 128f / 255f, 128f / 255f})
                .layout(ImageLayout.NHWC)
                .build();
        Tensor tensor = pipeline.transform(image);
        float[] data = tensor.toFloats();

        float expected = (1.0f - 127f / 255f) / (128f / 255f);
        assertEquals(expected, data[0], 0.01f);
    }

    private static BufferedImage createTestImage(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                image.setRGB(x, y, (x * 3) << 16 | (y * 2) << 8 | ((x + y) & 0xFF));
            }
        }
        return image;
    }
}
