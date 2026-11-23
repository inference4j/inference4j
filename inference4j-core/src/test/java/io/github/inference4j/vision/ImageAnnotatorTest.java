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

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class ImageAnnotatorTest {

    private static final int BLACK = 0x000000;
    private static final int GREEN = 0x00FF00;

    @Test
    void annotatePosesDrawsJointsAndBonesOnACopy() {
        BufferedImage image = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        Pose pose = pose(CocoJoint.LEFT_SHOULDER, 20, 50, CocoJoint.RIGHT_SHOULDER, 80, 50);

        BufferedImage annotated = ImageAnnotator.annotatePoses(image, List.of(pose));

        assertThat(rgb(annotated, 20, 50)).isEqualTo(GREEN);
        assertThat(rgb(annotated, 80, 50)).isEqualTo(GREEN);
        assertThat(rgb(annotated, 50, 50)).isEqualTo(GREEN);      // the shoulder bone
        assertThat(rgb(annotated, 50, 20)).isEqualTo(BLACK);
        assertThat(rgb(image, 20, 50)).isEqualTo(BLACK);          // source untouched
    }

    @Test
    void annotatePosesSkipsJointsBelowTheMinimumConfidenceAndTheirBones() {
        BufferedImage image = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        Pose pose = pose(CocoJoint.LEFT_SHOULDER, 20, 50, CocoJoint.RIGHT_SHOULDER, 80, 50);

        BufferedImage annotated = ImageAnnotator.annotatePoses(image, List.of(pose), CocoJoint.edges(), 0.95f,
                Color.RED, 2f);

        assertThat(rgb(annotated, 20, 50)).isEqualTo(0xFF0000);
        assertThat(rgb(annotated, 80, 50)).isEqualTo(BLACK);      // confidence 0.9
        assertThat(rgb(annotated, 50, 50)).isEqualTo(BLACK);
    }

    @Test
    void annotatePosesDrawsOnlyTheGivenBones() {
        BufferedImage image = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        Pose pose = pose(CocoJoint.LEFT_SHOULDER, 20, 50, CocoJoint.RIGHT_SHOULDER, 80, 50);

        BufferedImage annotated = ImageAnnotator.annotatePoses(image, List.of(pose), List.of(), 0.3f,
                Color.GREEN, 2f);

        assertThat(rgb(annotated, 20, 50)).isEqualTo(GREEN);
        assertThat(rgb(annotated, 50, 50)).isEqualTo(BLACK);
    }

    @Test
    void annotatePosesKeepsTheImageSize() {
        BufferedImage image = new BufferedImage(64, 48, BufferedImage.TYPE_3BYTE_BGR);

        BufferedImage annotated = ImageAnnotator.annotatePoses(image, List.of());

        assertThat(annotated.getWidth()).isEqualTo(64);
        assertThat(annotated.getHeight()).isEqualTo(48);
        assertThat(annotated).isNotSameAs(image);
    }

    /**
     * A pose with the two given joints visible (confidence 1.0 and 0.9) and every other joint off-screen at
     * confidence 0.1.
     */
    private static Pose pose(CocoJoint a, float ax, float ay, CocoJoint b, float bx, float by) {
        List<Keypoint> keypoints = new ArrayList<>();
        for (CocoJoint joint : CocoJoint.values()) {
            if (joint == a) {
                keypoints.add(new Keypoint(joint, ax, ay, 1.0f));
            } else if (joint == b) {
                keypoints.add(new Keypoint(joint, bx, by, 0.9f));
            } else {
                keypoints.add(new Keypoint(joint, 5, 5, 0.1f));
            }
        }
        return new Pose(new Detection(new BoundingBox(10, 10, 90, 90), "person", 0, 0.8f), keypoints);
    }

    private static int rgb(BufferedImage image, int x, int y) {
        return image.getRGB(x, y) & 0xFFFFFF;
    }
}
