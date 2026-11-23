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

import io.github.inference4j.Detector;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

/**
 * Finds people in images and the position of their body joints (pose estimation).
 *
 * <pre>{@code
 * try (PoseDetector detector = RtmoPoseDetector.builder().build()) {
 *     for (Pose pose : detector.detect(Path.of("gym.jpg"))) {
 *         float knee = pose.angle(CocoJoint.LEFT_HIP, CocoJoint.LEFT_KNEE, CocoJoint.LEFT_ANKLE);
 *     }
 * }
 * }</pre>
 *
 * <h2>Swap contract</h2>
 * <ul>
 *   <li>one {@link Pose} per person, sorted by descending person confidence;</li>
 *   <li>the person box is in pixel coordinates of the original image, inside the image, labeled {@code "person"};</li>
 *   <li>keypoints are in pixel coordinates of the original image and may fall outside it;</li>
 *   <li>every pose has every joint of the model's format, in its order ({@link CocoJoint} for COCO models); only
 *       the person confidence filters, never a keypoint's;</li>
 *   <li>{@code Path} and {@link BufferedImage} inputs give the same poses for the same image.</li>
 * </ul>
 *
 * @see Pose
 * @see ImageAnnotator#annotatePoses(BufferedImage, List)
 */
public interface PoseDetector extends Detector<BufferedImage, Pose> {

    /** Finds the poses of the people in the image, with the configured person confidence threshold. */
    @Override
    List<Pose> detect(BufferedImage image);

    /** Finds poses with a person confidence threshold other than the configured one. */
    List<Pose> detect(BufferedImage image, float confidenceThreshold);

    /** Reads the image file and finds poses with the configured person confidence threshold. */
    default List<Pose> detect(Path imagePath) {
        return detect(ImageFiles.read(imagePath));
    }

    /** Reads the image file and finds poses with a person confidence threshold other than the configured one. */
    default List<Pose> detect(Path imagePath, float confidenceThreshold) {
        return detect(ImageFiles.read(imagePath), confidenceThreshold);
    }

    @Override
    void close();
}
