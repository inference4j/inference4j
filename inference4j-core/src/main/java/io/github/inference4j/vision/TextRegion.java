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

import java.util.List;

/**
 * A region of an image that contains text.
 *
 * @param box        the axis-aligned box around the region
 * @param polygon    the region's outline: four corners clockwise from the top-left for detectors that find
 *                   rotated text, otherwise the corners of {@code box}
 * @param confidence the detector's confidence that the region contains text
 */
public record TextRegion(BoundingBox box, List<Point> polygon, float confidence) {

    public TextRegion {
        polygon = List.copyOf(polygon);
    }

    /** A region whose outline is its axis-aligned box. */
    public TextRegion(BoundingBox box, float confidence) {
        this(box, corners(box), confidence);
    }

    private static List<Point> corners(BoundingBox box) {
        return List.of(new Point(box.x1(), box.y1()), new Point(box.x2(), box.y1()),
                new Point(box.x2(), box.y2()), new Point(box.x1(), box.y2()));
    }
}
