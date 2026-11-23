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
import java.util.Objects;

/**
 * One person found by a {@link PoseDetector}: the person's box and score, and the position of each body joint.
 *
 * <pre>{@code
 * Keypoint wrist = pose.keypoint(CocoJoint.LEFT_WRIST);
 * float knee = pose.angle(CocoJoint.LEFT_HIP, CocoJoint.LEFT_KNEE, CocoJoint.LEFT_ANKLE);
 * }</pre>
 *
 * @param person    the person's box and score, with the label {@code "person"}
 * @param keypoints every joint of the model's format, in its order, whatever their confidence
 */
public record Pose(Detection person, List<Keypoint> keypoints) {

    public Pose {
        Objects.requireNonNull(person, "person");
        keypoints = List.copyOf(keypoints);
    }

    /**
     * The keypoint of a joint.
     *
     * @throws IllegalArgumentException if the pose has no such joint, for example a joint of another format
     */
    public Keypoint keypoint(Joint joint) {
        Keypoint keypoint = find(joint);
        if (keypoint == null) {
            throw new IllegalArgumentException("This pose has no joint " + joint.name());
        }
        return keypoint;
    }

    /** Whether the pose has a keypoint for this joint. */
    public boolean has(Joint joint) {
        return find(joint) != null;
    }

    /**
     * The angle at joint {@code b}, in degrees from 0 to 180, between the lines to {@code a} and to {@code c}: for
     * example the knee angle is {@code angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)}. Ignores confidence; check it on the
     * three keypoints first. Returns {@code NaN} when {@code b} is at the same position as {@code a} or {@code c}.
     *
     * @throws IllegalArgumentException if the pose has no keypoint for one of the joints
     */
    public float angle(Joint a, Joint b, Joint c) {
        Keypoint pa = keypoint(a);
        Keypoint pb = keypoint(b);
        Keypoint pc = keypoint(c);
        double ax = pa.x() - pb.x();
        double ay = pa.y() - pb.y();
        double cx = pc.x() - pb.x();
        double cy = pc.y() - pb.y();
        if ((ax == 0 && ay == 0) || (cx == 0 && cy == 0)) {
            return Float.NaN;
        }
        return (float) Math.toDegrees(Math.abs(Math.atan2(ax * cy - ay * cx, ax * cx + ay * cy)));
    }

    private Keypoint find(Joint joint) {
        Objects.requireNonNull(joint, "joint");
        if (joint.index() >= 0 && joint.index() < keypoints.size()
                && keypoints.get(joint.index()).joint().equals(joint)) {
            return keypoints.get(joint.index());
        }
        for (Keypoint keypoint : keypoints) {
            if (keypoint.joint().equals(joint)) {
                return keypoint;
            }
        }
        return null;
    }
}
