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
 * The 17 body keypoints of COCO, in the order the models output them. "Left" and "right" are the person's own
 * left and right, not the viewer's.
 *
 * @see Pose#keypoint(Joint)
 */
public enum CocoJoint implements Joint {

    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE;

    private static final List<Joint.Edge> EDGES = List.of(
            edge(LEFT_ANKLE, LEFT_KNEE), edge(LEFT_KNEE, LEFT_HIP),
            edge(RIGHT_ANKLE, RIGHT_KNEE), edge(RIGHT_KNEE, RIGHT_HIP),
            edge(LEFT_HIP, RIGHT_HIP),
            edge(LEFT_SHOULDER, LEFT_HIP), edge(RIGHT_SHOULDER, RIGHT_HIP),
            edge(LEFT_SHOULDER, RIGHT_SHOULDER),
            edge(LEFT_SHOULDER, LEFT_ELBOW), edge(RIGHT_SHOULDER, RIGHT_ELBOW),
            edge(LEFT_ELBOW, LEFT_WRIST), edge(RIGHT_ELBOW, RIGHT_WRIST),
            edge(LEFT_EYE, RIGHT_EYE), edge(NOSE, LEFT_EYE), edge(NOSE, RIGHT_EYE),
            edge(LEFT_EYE, LEFT_EAR), edge(RIGHT_EYE, RIGHT_EAR),
            edge(LEFT_EAR, LEFT_SHOULDER), edge(RIGHT_EAR, RIGHT_SHOULDER));

    private static final List<CocoJoint> ALL = List.of(values());

    /** The same as {@link #ordinal()}: COCO keypoint order. */
    @Override
    public int index() {
        return ordinal();
    }

    /** The 19 bones of the COCO skeleton, as drawn by the COCO tools and MMPose. */
    public static List<Joint.Edge> edges() {
        return EDGES;
    }

    /**
     * The joint at a COCO keypoint index.
     *
     * @throws IllegalArgumentException if {@code index} is not between 0 and 16
     */
    public static CocoJoint of(int index) {
        if (index < 0 || index >= ALL.size()) {
            throw new IllegalArgumentException("COCO joint index must be between 0 and 16, got " + index);
        }
        return ALL.get(index);
    }

    private static Joint.Edge edge(CocoJoint from, CocoJoint to) {
        return new Joint.Edge(from, to);
    }
}
