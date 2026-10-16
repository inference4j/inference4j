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

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TextRegionTest {

    @Test
    void aRegionBuiltFromABoxUsesTheBoxCornersAsItsOutline() {
        TextRegion region = new TextRegion(new BoundingBox(1, 2, 11, 7), 0.9f);

        assertThat(region.polygon()).containsExactly(
                new Point(1, 2), new Point(11, 2), new Point(11, 7), new Point(1, 7));
        assertThat(region.confidence()).isEqualTo(0.9f);
    }

    @Test
    void theOutlineCannotBeChangedAfterwards() {
        List<Point> outline = new ArrayList<>(List.of(new Point(0, 0), new Point(4, 0), new Point(4, 2)));
        TextRegion region = new TextRegion(new BoundingBox(0, 0, 4, 2), outline, 1f);
        outline.clear();

        assertThat(region.polygon()).hasSize(3);
        assertThatThrownBy(() -> region.polygon().add(new Point(9, 9)))
                .isInstanceOf(UnsupportedOperationException.class);
    }
}
