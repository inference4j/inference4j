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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Sorts text regions top to bottom, then left to right, as PaddleOCR's {@code SortQuadBoxes} does: by the first
 * polygon point, treating regions whose tops are less than 10 pixels apart as one line.
 */
final class ReadingOrder {

    private static final float SAME_LINE_PIXELS = 10f;

    private ReadingOrder() {
    }

    static List<TextRegion> sort(List<TextRegion> regions) {
        List<TextRegion> sorted = new ArrayList<>(regions);
        sorted.sort(Comparator.comparingDouble((TextRegion r) -> first(r).y()).thenComparingDouble(r -> first(r).x()));
        for (int i = 0; i < sorted.size() - 1; i++) {
            for (int j = i; j >= 0; j--) {
                Point next = first(sorted.get(j + 1));
                Point current = first(sorted.get(j));
                if (Math.abs(next.y() - current.y()) < SAME_LINE_PIXELS && next.x() < current.x()) {
                    TextRegion swap = sorted.get(j);
                    sorted.set(j, sorted.get(j + 1));
                    sorted.set(j + 1, swap);
                } else {
                    break;
                }
            }
        }
        return sorted;
    }

    private static Point first(TextRegion region) {
        return region.polygon().get(0);
    }
}
