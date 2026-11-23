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
import static org.assertj.core.api.Assertions.within;

class PoseTest {

    private static final Detection PERSON = new Detection(new BoundingBox(0, 0, 100, 200), "person", 0, 0.9f);

    @Test
    void cocoJointsFollowTheCocoKeypointOrder() {
        assertThat(CocoJoint.values()).hasSize(17);
        assertThat(CocoJoint.NOSE.index()).isZero();
        assertThat(CocoJoint.LEFT_SHOULDER.index()).isEqualTo(5);
        assertThat(CocoJoint.LEFT_HIP.index()).isEqualTo(11);
        assertThat(CocoJoint.RIGHT_ANKLE.index()).isEqualTo(16);
        assertThat(CocoJoint.of(9)).isEqualTo(CocoJoint.LEFT_WRIST);
    }

    @Test
    void cocoJointRejectsAnIndexOutsideTheFormat() {
        assertThatThrownBy(() -> CocoJoint.of(17)).isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> CocoJoint.of(-1)).isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void cocoSkeletonHasNineteenBonesTouchingEveryJoint() {
        List<Joint.Edge> edges = CocoJoint.edges();

        assertThat(edges).hasSize(19).doesNotHaveDuplicates();
        assertThat(edges).contains(new Joint.Edge(CocoJoint.LEFT_KNEE, CocoJoint.LEFT_HIP),
                new Joint.Edge(CocoJoint.LEFT_SHOULDER, CocoJoint.RIGHT_SHOULDER));
        List<Joint> touched = new ArrayList<>();
        edges.forEach(edge -> {
            touched.add(edge.from());
            touched.add(edge.to());
        });
        assertThat(touched).containsAll(List.of(CocoJoint.values()));
    }

    @Test
    void keypointFindsTheJoint() {
        Pose pose = pose();

        assertThat(pose.keypoint(CocoJoint.LEFT_KNEE).x()).isEqualTo(13f);
        assertThat(pose.keypoint(CocoJoint.LEFT_KNEE).joint()).isEqualTo(CocoJoint.LEFT_KNEE);
        assertThat(pose.keypoint(CocoJoint.NOSE).point()).isEqualTo(new Point(0, 100));
    }

    @Test
    void keypointRejectsAJointThePoseDoesNotHave() {
        Joint other = new Joint() {
            public int index() {
                return 0;
            }

            public String name() {
                return "TAIL";
            }
        };
        Pose pose = pose();

        assertThat(pose.has(other)).isFalse();
        assertThatThrownBy(() -> pose.keypoint(other))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("TAIL");
    }

    @Test
    void keypointsAreCopiedAndUnmodifiable() {
        List<Keypoint> keypoints = new ArrayList<>(pose().keypoints());
        Pose pose = new Pose(PERSON, keypoints);

        keypoints.clear();

        assertThat(pose.keypoints()).hasSize(17);
        assertThatThrownBy(() -> pose.keypoints().clear()).isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void angleIsMeasuredAtTheMiddleJoint() {
        Pose bent = withPoints(CocoJoint.LEFT_HIP, 0, 0, CocoJoint.LEFT_KNEE, 0, 10, CocoJoint.LEFT_ANKLE, 10, 10);
        Pose straight = withPoints(CocoJoint.LEFT_HIP, 0, 0, CocoJoint.LEFT_KNEE, 0, 10, CocoJoint.LEFT_ANKLE, 0, 20);
        Pose folded = withPoints(CocoJoint.LEFT_HIP, 0, 0, CocoJoint.LEFT_KNEE, 0, 10, CocoJoint.LEFT_ANKLE, 1, 0);

        assertThat(bent.angle(CocoJoint.LEFT_HIP, CocoJoint.LEFT_KNEE, CocoJoint.LEFT_ANKLE))
                .isCloseTo(90f, within(1e-4f));
        assertThat(straight.angle(CocoJoint.LEFT_HIP, CocoJoint.LEFT_KNEE, CocoJoint.LEFT_ANKLE))
                .isCloseTo(180f, within(1e-4f));
        assertThat(folded.angle(CocoJoint.LEFT_HIP, CocoJoint.LEFT_KNEE, CocoJoint.LEFT_ANKLE))
                .isCloseTo(5.71f, within(1e-2f));
        assertThat(folded.angle(CocoJoint.LEFT_ANKLE, CocoJoint.LEFT_KNEE, CocoJoint.LEFT_HIP))
                .isCloseTo(5.71f, within(1e-2f));
    }

    @Test
    void angleIsNaNWhenTwoJointsCoincide() {
        Pose pose = withPoints(CocoJoint.LEFT_HIP, 5, 5, CocoJoint.LEFT_KNEE, 5, 5, CocoJoint.LEFT_ANKLE, 10, 10);

        assertThat(pose.angle(CocoJoint.LEFT_HIP, CocoJoint.LEFT_KNEE, CocoJoint.LEFT_ANKLE)).isNaN();
    }

    /** Joint i at (i, 100 + i), confidence 0.5. */
    private static Pose pose() {
        List<Keypoint> keypoints = new ArrayList<>();
        for (CocoJoint joint : CocoJoint.values()) {
            keypoints.add(new Keypoint(joint, joint.index(), 100 + joint.index(), 0.5f));
        }
        return new Pose(PERSON, keypoints);
    }

    private static Pose withPoints(CocoJoint a, float ax, float ay, CocoJoint b, float bx, float by,
                                   CocoJoint c, float cx, float cy) {
        List<Keypoint> keypoints = new ArrayList<>(pose().keypoints());
        keypoints.set(a.index(), new Keypoint(a, ax, ay, 0.5f));
        keypoints.set(b.index(), new Keypoint(b, bx, by, 0.5f));
        keypoints.set(c.index(), new Keypoint(c, cx, cy, 0.5f));
        return new Pose(PERSON, keypoints);
    }
}
