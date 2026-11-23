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

import io.github.inference4j.TestModels;
import io.github.inference4j.exception.ModelLoadException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.ModelSource;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * Runs the detector on the {@code rtmo-people} fixture: four people and a padding row on a 32 x 32 input, see {@code generate.py}.
 * A 64 x 64 image is scaled by 0.5 into that input, so every coordinate comes back doubled.
 */
class RtmoPoseDetectorTest {

    @TempDir
    Path dir;

    @Test
    void detectReturnsConfidentPeopleMostConfidentFirst() {
        try (PoseDetector detector = build(RtmoPoseDetector.builder())) {
            List<Pose> poses = detector.detect(image(64, 64));

            assertThat(poses).hasSize(2);
            assertThat(poses.get(0).person()).isEqualTo(new Detection(new BoundingBox(4, 4, 28, 56), "person", 0, 0.9f));
            assertThat(poses.get(1).person()).isEqualTo(new Detection(new BoundingBox(36, 8, 60, 56), "person", 0, 0.8f));
        }
    }

    @Test
    void keypointsAreMappedBackToTheImageWhateverTheirConfidence() {
        try (PoseDetector detector = build(RtmoPoseDetector.builder())) {
            Pose first = detector.detect(image(64, 64)).get(0);     // fixture row 1

            assertThat(first.keypoints()).hasSize(17);
            assertThat(first.keypoints()).extracting(Keypoint::joint).containsExactly(CocoJoint.values());
            Keypoint nose = first.keypoint(CocoJoint.NOSE);
            assertThat(nose.x()).isCloseTo(8f, within(1e-4f));
            assertThat(nose.y()).isCloseTo(6f, within(1e-4f));
            assertThat(nose.confidence()).isCloseTo(0.9f, within(1e-6f));
            Keypoint leftEye = first.keypoint(CocoJoint.LEFT_EYE);
            assertThat(leftEye.x()).isCloseTo(9f, within(1e-4f));
            assertThat(leftEye.y()).isCloseTo(8f, within(1e-4f));
            assertThat(leftEye.confidence()).isCloseTo(0.1f, within(1e-6f));
        }
    }

    @Test
    void lowerThresholdPerCallKeepsLessConfidentPeople() {
        try (PoseDetector detector = build(RtmoPoseDetector.builder())) {
            List<Pose> poses = detector.detect(image(64, 64), 0.1f);

            assertThat(poses).extracting(pose -> pose.person().confidence()).containsExactly(0.9f, 0.8f, 0.2f);
        }
    }

    @Test
    void paddingRowNeverBecomesAPersonWhateverTheThreshold() {
        try (PoseDetector detector = build(RtmoPoseDetector.builder().confidenceThreshold(0f).iouThreshold(1f))) {
            assertThat(detector.detect(image(64, 64), 0f)).extracting(pose -> pose.person().confidence())
                    .containsExactly(0.9f, 0.8f, 0.7f, 0.2f);
        }
    }

    @Test
    void builderThresholdsApply() {
        try (PoseDetector strict = build(RtmoPoseDetector.builder().confidenceThreshold(0.85f));
             PoseDetector noNms = build(RtmoPoseDetector.builder().iouThreshold(1f));
             PoseDetector single = build(RtmoPoseDetector.builder().maxPoses(1))) {
            assertThat(strict.detect(image(64, 64))).hasSize(1);
            assertThat(noNms.detect(image(64, 64))).extracting(pose -> pose.person().confidence())
                    .containsExactly(0.9f, 0.8f, 0.7f);
            assertThat(single.detect(image(64, 64))).hasSize(1);
        }
    }

    @Test
    void boxesAreClampedToTheImageButKeypointsAreNot() {
        // A 64 x 32 image is also scaled by 0.5; the second person's box (36..60, 8..56) is cut at the bottom edge
        try (PoseDetector detector = build(RtmoPoseDetector.builder())) {
            List<Pose> poses = detector.detect(image(64, 32));

            assertThat(poses.get(1).person().box()).isEqualTo(new BoundingBox(36, 8, 60, 32));
            assertThat(poses.get(1).keypoint(CocoJoint.RIGHT_ANKLE).y()).isCloseTo(38f, within(1e-4f));
        }
    }

    @Test
    void pathAndImageGiveTheSamePoses(@TempDir Path images) throws Exception {
        Path file = images.resolve("people.png");
        ImageAnnotator.save(image(64, 64), images.resolve("people.png"));
        try (PoseDetector detector = build(RtmoPoseDetector.builder())) {
            assertThat(detector.detect(file)).isEqualTo(detector.detect(image(64, 64)));
            assertThat(detector.detect(file, 0.1f)).hasSize(3);
        }
    }

    @Test
    void inputSizeOtherThanTheModelsFixedSizeFails() {
        assertThatThrownBy(() -> build(RtmoPoseDetector.builder().inputSize(640)))
                .isInstanceOf(ModelLoadException.class)
                .hasMessageContaining("fixed 32 × 32");
    }

    @Test
    void maxPosesMustBePositive() {
        assertThatThrownBy(() -> RtmoPoseDetector.builder().maxPoses(0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void closedDetectorRejectsDetect() {
        PoseDetector detector = build(RtmoPoseDetector.builder());

        detector.close();

        assertThatThrownBy(() -> detector.detect(image(64, 64))).isInstanceOf(IllegalStateException.class);
    }

    @Test
    void presetsLoadTheirModelIds() {
        assertThat(requestedId(RtmoPoseDetector.builder())).isEqualTo("inference4j/rtmo-s-coco");
        assertThat(requestedId(RtmoPoseDetector.s())).isEqualTo("inference4j/rtmo-s-coco");
        assertThat(requestedId(RtmoPoseDetector.m())).isEqualTo("inference4j/rtmo-m-coco");
        assertThat(requestedId(RtmoPoseDetector.l())).isEqualTo("inference4j/rtmo-l-coco");
    }

    private RtmoPoseDetector build(RtmoPoseDetector.Builder builder) {
        return builder.modelSource(TestModels.model(dir, "rtmo-people")).build();
    }

    private static String requestedId(RtmoPoseDetector.Builder builder) {
        String[] requested = new String[1];
        ModelSource missing = id -> {
            requested[0] = id;
            return Path.of("/nonexistent/path/" + id);
        };
        assertThatThrownBy(() -> builder.modelSource(missing).build()).isInstanceOf(ModelSourceException.class);
        return requested[0];
    }

    private static BufferedImage image(int width, int height) {
        return new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    }
}
