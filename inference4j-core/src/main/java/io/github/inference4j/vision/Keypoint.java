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

import java.util.Objects;

/**
 * One body joint of a {@link Pose}.
 *
 * <p>Coordinates are in pixels of the original image. A joint the model cannot see (hidden, or outside the image)
 * is still returned, with a low confidence and a guessed position that may fall outside the image; filter on
 * {@code confidence} where that matters.
 *
 * @param joint      which joint this is
 * @param x          horizontal position (pixels, from the left edge)
 * @param y          vertical position (pixels, from the top edge)
 * @param confidence the model's score for the joint, from 0 to 1
 */
public record Keypoint(Joint joint, float x, float y, float confidence) {

    public Keypoint {
        Objects.requireNonNull(joint, "joint");
    }

    /** The position as a {@link Point}. */
    public Point point() {
        return new Point(x, y);
    }
}
