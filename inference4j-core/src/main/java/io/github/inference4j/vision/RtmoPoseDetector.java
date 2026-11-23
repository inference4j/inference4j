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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.model.AbstractModelBuilder;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Multi-person pose detector for <a href="https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo">RTMO</a>
 * (OpenMMLab MMPose), a one-stage model that finds every person and their 17 COCO keypoints in one pass, so its cost
 * does not grow with the number of people. Code and weights are Apache-2.0.
 *
 * <pre>{@code
 * try (PoseDetector detector = RtmoPoseDetector.builder().build()) {
 *     for (Pose pose : detector.detect(Path.of("gym.jpg"))) {
 *         float knee = pose.angle(CocoJoint.LEFT_HIP, CocoJoint.LEFT_KNEE, CocoJoint.LEFT_ANKLE);
 *     }
 * }
 * }</pre>
 *
 * <p>Presets, 640 × 640 input, the MMPose checkpoints trained on COCO only (not the "body7" checkpoints, which add
 * datasets with non-commercial terms):
 * <ul>
 *   <li>{@link #s()} (default): 67.7 COCO keypoint AP;</li>
 *   <li>{@link #m()}: 70.9 AP;</li>
 *   <li>{@link #l()}: 72.4 AP, the slowest preset.</li>
 * </ul>
 * AP figures are the authors' (COCO val2017); inference4j has not re-measured them.
 *
 * <p>Pre-processing matches rtmlib and {@link YoloXDetector}: the image is scaled to fit the input, keeping its
 * aspect ratio, with OpenCV's bilinear resize, placed at the top left and padded with gray (114). The input is BGR
 * with raw 0–255 values. The exported model decodes and runs non-maximum suppression itself (MMDeploy: score above 0.1, IoU 0.65,
 * at most 50 people) and returns person boxes with scores and, per person, 17 keypoints with scores. Persons below
 * the confidence threshold are then dropped, and a second suppression pass applies the configured IoU threshold
 * (a no-op at the default). Keypoints are never filtered by their own score. Thresholds at or below 0.1 have no
 * effect, since the model has already dropped those persons.
 *
 * @see PoseDetector
 * @see Pose
 */
public class RtmoPoseDetector implements PoseDetector {

    private static final String S = "inference4j/rtmo-s-coco";
    private static final String M = "inference4j/rtmo-m-coco";
    private static final String L = "inference4j/rtmo-l-coco";
    private static final String DETS = "dets";
    private static final String KEYPOINTS = "keypoints";
    private static final String PERSON = "person";
    private static final float PAD_VALUE = 114f;
    /** The exported model's own score threshold; its padding row scores 0, so it must never get through. */
    private static final float MODEL_SCORE_THRESHOLD = 0.1f;

    private final InferenceSession session;
    private final String inputName;
    private final int inputSize;
    private final float defaultConfidenceThreshold;
    private final float iouThreshold;
    private final int maxPoses;

    private RtmoPoseDetector(InferenceSession session, String inputName, int inputSize, float confidenceThreshold,
                             float iouThreshold, int maxPoses) {
        this.session = session;
        this.inputName = inputName;
        this.inputSize = inputSize;
        this.defaultConfidenceThreshold = confidenceThreshold;
        this.iouThreshold = iouThreshold;
        this.maxPoses = maxPoses;
    }

    /** RTMO-s, COCO only (640 × 640): 67.7 COCO keypoint AP (the authors' figure). The default. */
    public static Builder s() {
        return new Builder().modelId(S);
    }

    /** RTMO-m, COCO only (640 × 640): 70.9 COCO keypoint AP (the authors' figure). */
    public static Builder m() {
        return new Builder().modelId(M);
    }

    /** RTMO-l, COCO only (640 × 640): 72.4 COCO keypoint AP (the authors' figure), the slowest preset. */
    public static Builder l() {
        return new Builder().modelId(L);
    }

    /** A builder for the default model, {@link #s()}. */
    public static Builder builder() {
        return s();
    }

    @Override
    public List<Pose> detect(BufferedImage image) {
        return detect(image, defaultConfidenceThreshold);
    }

    @Override
    public List<Pose> detect(BufferedImage image, float confidenceThreshold) {
        int width = image.getWidth();
        int height = image.getHeight();
        DetectionOps.Letterbox letterbox = DetectionOps.Letterbox.topLeft(width, height, inputSize);
        Map<String, Tensor> outputs = session.run(Map.of(inputName,
                Tensor.fromFloats(toInput(image, letterbox), new long[]{1, 3, inputSize, inputSize})));
        Tensor dets = output(outputs, DETS, 3);
        Tensor keypoints = output(outputs, KEYPOINTS, 4);
        return decode(dets.toFloats(), dets.shape(), keypoints.toFloats(), keypoints.shape(), letterbox, width,
                height, confidenceThreshold);
    }

    @Override
    public void close() {
        session.close();
    }

    /** The image scaled into the top left of the input, the rest gray, as planar BGR with raw 0–255 values. */
    private float[] toInput(BufferedImage image, DetectionOps.Letterbox letterbox) {
        int scaledWidth = letterbox.scaledWidth();
        int scaledHeight = letterbox.scaledHeight();
        int[] scaled = PaddleImages.resize(PaddleImages.pixels(image), image.getWidth(), image.getHeight(),
                scaledWidth, scaledHeight);
        int plane = inputSize * inputSize;
        float[] data = new float[3 * plane];
        Arrays.fill(data, PAD_VALUE);
        for (int y = 0; y < scaledHeight; y++) {
            for (int x = 0; x < scaledWidth; x++) {
                int rgb = scaled[y * scaledWidth + x];
                int i = y * inputSize + x;
                data[i] = rgb & 0xFF;                          // blue
                data[plane + i] = (rgb >> 8) & 0xFF;           // green
                data[2 * plane + i] = (rgb >> 16) & 0xFF;      // red
            }
        }
        return data;
    }

    /** The output with the given name or, failing that, the only output of the given rank. */
    private static Tensor output(Map<String, Tensor> outputs, String name, int rank) {
        Tensor named = outputs.get(name);
        if (named != null) {
            return named;
        }
        List<Tensor> ofRank = outputs.values().stream().filter(t -> t.shape().length == rank).toList();
        if (ofRank.size() != 1) {
            throw new InferenceException("Expected an RTMO output '" + name + "' of rank " + rank + ", got outputs "
                    + outputs.keySet());
        }
        return ofRank.get(0);
    }

    /**
     * Decodes {@code dets [1, N, 5]} (x1, y1, x2, y2, score in input pixels) and {@code keypoints [1, N, K, 3]}
     * (x, y, score in input pixels). Fewer than 50 rows end with a padding row of score 0, dropped with every row at
     * or below the model's own threshold of 0.1, whatever the caller's threshold.
     */
    private List<Pose> decode(float[] dets, long[] detsShape, float[] keypoints, long[] keypointsShape,
                              DetectionOps.Letterbox letterbox, int width, int height, float confidenceThreshold) {
        if (detsShape.length != 3 || detsShape[2] != 5 || keypointsShape.length != 4 || keypointsShape[3] != 3
                || keypointsShape[1] != detsShape[1]) {
            throw new InferenceException("Expected RTMO outputs dets [1, N, 5] and keypoints [1, N, K, 3], got "
                    + Arrays.toString(detsShape) + " and " + Arrays.toString(keypointsShape));
        }
        int people = (int) detsShape[1];
        int joints = (int) keypointsShape[2];
        if (joints != CocoJoint.values().length) {
            throw new InferenceException("Expected " + CocoJoint.values().length + " COCO keypoints per person, got "
                    + joints);
        }
        List<Detection> candidates = new ArrayList<>();
        List<Integer> rows = new ArrayList<>();
        for (int p = 0; p < people; p++) {
            float score = dets[p * 5 + 4];
            if (score <= MODEL_SCORE_THRESHOLD || score < confidenceThreshold) {
                continue;
            }
            float[] box = {dets[p * 5], dets[p * 5 + 1], dets[p * 5 + 2], dets[p * 5 + 3]};
            // classIndex carries the row until NMS has run; it is reset to 0 below
            candidates.add(new Detection(letterbox.toOriginal(box, width, height), PERSON, p, score));
            rows.add(p);
        }
        candidates.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
        List<Detection> kept = DetectionOps.nms(candidates, iouThreshold, false, maxPoses);
        float scale = letterbox.scale();
        List<Pose> poses = new ArrayList<>(kept.size());
        for (Detection d : kept) {
            int p = d.classIndex();
            List<Keypoint> points = new ArrayList<>(joints);
            for (int k = 0; k < joints; k++) {
                int base = (p * joints + k) * 3;
                points.add(new Keypoint(CocoJoint.of(k), (keypoints[base] - letterbox.padX()) / scale,
                        (keypoints[base + 1] - letterbox.padY()) / scale, keypoints[base + 2]));
            }
            poses.add(new Pose(new Detection(d.box(), PERSON, 0, d.confidence()), points));
        }
        return poses;
    }

    public static class Builder extends AbstractModelBuilder<RtmoPoseDetector, Builder> {
        private Integer inputSize;
        private float confidenceThreshold = 0.5f;
        private float iouThreshold = 0.65f;
        private int maxPoses = 50;

        public Builder() {
            super(S);
        }

        /**
         * Side of the square model input, in pixels. A model with a fixed input size uses that size, and a different
         * value fails at {@link #build()}; otherwise defaults to 640.
         */
        public Builder inputSize(int inputSize) {
            this.inputSize = inputSize;
            return this;
        }

        /** Persons scoring below this are dropped. Keypoint scores never drop a person. Defaults to 0.5. */
        public Builder confidenceThreshold(float confidenceThreshold) {
            this.confidenceThreshold = confidenceThreshold;
            return this;
        }

        /**
         * IoU above which the box of a less confident person is removed. Defaults to 0.65, the value the exported
         * model already applies; only a lower value changes the result.
         */
        public Builder iouThreshold(float iouThreshold) {
            this.iouThreshold = iouThreshold;
            return this;
        }

        /** The most poses returned per image, the most confident first. Defaults to 50, the most the exported models return. */
        public Builder maxPoses(int maxPoses) {
            if (maxPoses < 1) {
                throw new IllegalArgumentException("maxPoses must be at least 1, got " + maxPoses);
            }
            this.maxPoses = maxPoses;
            return this;
        }

        public RtmoPoseDetector build() {
            String id = resolvedModelId();
            return load(List.of("model.onnx"), files -> {
                InferenceSession session = files.session("model.onnx");
                String inputName = session.inputNames().iterator().next();
                int size = DetectorModels.inputSize(inputSize,
                        DetectorModels.fixedSize(session.inputShape(inputName)), id);
                return new RtmoPoseDetector(session, inputName, size, confidenceThreshold, iouThreshold, maxPoses);
            });
        }
    }
}
