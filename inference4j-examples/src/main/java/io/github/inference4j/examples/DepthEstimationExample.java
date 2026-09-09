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

package io.github.inference4j.examples;

import java.nio.file.Path;

import io.github.inference4j.vision.Colormap;
import io.github.inference4j.vision.DepthAnythingEstimator;
import io.github.inference4j.vision.DepthMap;
import io.github.inference4j.vision.ImageAnnotator;

/**
 * Demonstrates monocular depth estimation with Depth Anything V2.
 *
 * Predicts how far every pixel is from the camera using a single ordinary photo,
 * then writes the result out as both a grayscale and a colormapped image.
 *
 * Run with: ./gradlew :inference4j-examples:run -PmainClass=io.github.inference4j.examples.DepthEstimationExample
 */
public class DepthEstimationExample {

    public static void main(String[] args) throws Exception {
        Path imagePath = Path.of(
                DepthEstimationExample.class.getResource("/fixtures/sample.jpg").toURI());

        System.out.println("=== Depth Estimation (Depth Anything V2) ===");
        try (DepthAnythingEstimator estimator = DepthAnythingEstimator.builder().build()) {
            DepthMap depth = estimator.estimate(imagePath);

            System.out.printf("Image size:  %d x %d%n", depth.width(), depth.height());
            System.out.printf("Depth range: %.3f (furthest) to %.3f (nearest)%n",
                    depth.min(), depth.max());

            // Values are relative inverse depth — larger means closer to the camera.
            // The scale is model-dependent, so compare within one map, never across two.
            int cx = depth.width() / 2;
            int cy = depth.height() / 2;
            System.out.printf("Centre pixel (%d, %d): %.3f%n", cx, cy, depth.at(cx, cy));
            System.out.printf("Top-left corner:      %.3f%n", depth.at(0, 0));

            System.out.println();
            System.out.println("=== Writing depth images ===");

            // Grayscale is fine for data, but a perceptual colormap makes near/far
            // far easier to read at a glance.
            Path grayPath = sibling(imagePath, "_depth_gray.png");
            ImageAnnotator.save(depth.toImage(), grayPath);
            System.out.printf("Grayscale saved to %s%n", grayPath);

            Path turboPath = sibling(imagePath, "_depth_turbo.png");
            ImageAnnotator.save(depth.toImage(Colormap.TURBO), turboPath);
            System.out.printf("Turbo colormap saved to %s%n", turboPath);

            Path viridisPath = sibling(imagePath, "_depth_viridis.png");
            ImageAnnotator.save(depth.toImage(Colormap.VIRIDIS), viridisPath);
            System.out.printf("Viridis colormap saved to %s%n", viridisPath);
        }
    }

    private static Path sibling(Path original, String suffix) {
        String filename = original.getFileName().toString();
        int dot = filename.lastIndexOf('.');
        String name = dot > 0 ? filename.substring(0, dot) : filename;
        return original.getParent().resolve(name + suffix);
    }
}
