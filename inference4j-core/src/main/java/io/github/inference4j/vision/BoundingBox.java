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

/**
 * An axis-aligned bounding box in pixel coordinates.
 *
 * <p>Coordinates are in {@code (x1, y1, x2, y2)} format where
 * {@code (x1, y1)} is the top-left corner and {@code (x2, y2)} is
 * the bottom-right corner, both in original image pixels.
 *
 * @param x1 left edge (pixels)
 * @param y1 top edge (pixels)
 * @param x2 right edge (pixels)
 * @param y2 bottom edge (pixels)
 */
public record BoundingBox(float x1, float y1, float x2, float y2) {

    /** Width of the bounding box in pixels. */
    public float width() {
        return x2 - x1;
    }

    /** Height of the bounding box in pixels. */
    public float height() {
        return y2 - y1;
    }

    /** Area of the bounding box in square pixels. */
    public float area() {
        return width() * height();
    }
}
