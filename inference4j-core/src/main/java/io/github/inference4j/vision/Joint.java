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
 * A body joint in a keypoint format, such as {@link CocoJoint}. Each format is an enum implementing this interface,
 * so {@link Pose#keypoint(Joint)} stays type-safe and new formats can be added without changing {@link Pose}.
 *
 * @see CocoJoint
 * @see Keypoint
 */
public interface Joint {

    /** Position of the joint in the model output, from 0. */
    int index();

    /** Name of the joint, such as {@code LEFT_KNEE}. */
    String name();

    /**
     * A bone of the skeleton: the line drawn between two joints.
     *
     * @param from one end
     * @param to   the other end
     */
    record Edge(Joint from, Joint to) {

        public Edge {
            Objects.requireNonNull(from, "from");
            Objects.requireNonNull(to, "to");
        }
    }
}
