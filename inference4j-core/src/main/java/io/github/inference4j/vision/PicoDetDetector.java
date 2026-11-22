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
import io.github.inference4j.preprocessing.image.Labels;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Object detector for <a href="https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/picodet">
 * PP-PicoDet</a> (PaddleDetection), a small anchor-free CNN detector for CPUs. Code and weights are Apache-2.0.
 *
 * <pre>{@code
 * try (ObjectDetector detector = PicoDetDetector.s().build()) {
 *     List<Detection> detections = detector.detect(Path.of("street.jpg"));
 * }
 * }</pre>
 *
 * <p>Presets, trained on COCO by the PaddleDetection authors, 416 × 416 input, from their official ONNX export
 * without post-processing:
 * <ul>
 *   <li>{@link #s()} (default): 32.5 COCO AP, 4.6 MB;</li>
 *   <li>{@link #m()}: 37.5 COCO AP, 13 MB.</li>
 * </ul>
 *
 * <p>Matches the model's PaddleDetection configuration: the image is stretched to the input with OpenCV's bicubic
 * resize, scaled to [0, 1] and normalized with the ImageNet mean and standard deviation (RGB). The model outputs
 * final class scores and, per side of each box, a distribution over 8 distances; the expected distance places the
 * box around its anchor point (strides 8, 16, 32 and 64). Every class above the threshold counts, so one anchor can
 * give two classes, and overlapping boxes of the same class are removed with non-maximum suppression.
 *
 * <p>Labels default to {@link Labels#coco()}, the same as the other detectors.
 *
 * @see ObjectDetector
 * @see Detection
 */
public class PicoDetDetector implements ObjectDetector {

    private static final String S = "inference4j/picodet-s";
    private static final String M = "inference4j/picodet-m";
    private static final Set<String> PRESETS = Set.of(S, M);
    private static final int[] STRIDES = {8, 16, 32, 64};
    private static final int BINS = 8;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private final InferenceSession session;
    private final String inputName;
    private final Labels labels;
    private final int inputSize;
    private final float defaultConfidenceThreshold;
    private final float defaultIouThreshold;
    private final int maxDetections;

    private PicoDetDetector(InferenceSession session, String inputName, Labels labels, int inputSize,
                            float confidenceThreshold, float iouThreshold, int maxDetections) {
        this.session = session;
        this.inputName = inputName;
        this.labels = labels;
        this.inputSize = inputSize;
        this.defaultConfidenceThreshold = confidenceThreshold;
        this.defaultIouThreshold = iouThreshold;
        this.maxDetections = maxDetections;
    }

    /** PP-PicoDet S (4.6 MB): 32.5 COCO AP (the authors' figure). The default. */
    public static Builder s() {
        return new Builder().modelId(S);
    }

    /** PP-PicoDet M (13 MB): 37.5 COCO AP (the authors' figure), slower than S. */
    public static Builder m() {
        return new Builder().modelId(M);
    }

    /** A builder for the default model, {@link #s()}. */
    public static Builder builder() {
        return s();
    }

    @Override
    public List<Detection> detect(BufferedImage image) {
        return detect(image, defaultConfidenceThreshold, defaultIouThreshold);
    }

    /** Detects objects with a confidence threshold other than the configured one, and the configured IoU threshold. */
    @Override
    public List<Detection> detect(BufferedImage image, float confidenceThreshold) {
        return detect(image, confidenceThreshold, defaultIouThreshold);
    }

    @Override
    public List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold) {
        int width = image.getWidth();
        int height = image.getHeight();
        int[] resized = PaddleImages.resizeCubic(PaddleImages.pixels(image), width, height, inputSize, inputSize);
        float[] input = DetectionOps.toChw(resized, inputSize, inputSize, MEAN, STD, false);
        Map<String, Tensor> outputs = session.run(Map.of(inputName,
                Tensor.fromFloats(input, new long[]{1, 3, inputSize, inputSize})));
        return decode(new ArrayList<>(outputs.values()), width, height, confidenceThreshold, iouThreshold);
    }

    @Override
    public void close() {
        session.close();
    }

    /**
     * PaddleDetection's {@code PicoHeadV2} decode and {@code MultiClassNMS}: outputs are the class scores per stride,
     * {@code [1, anchors, classes]}, then the box distributions per stride, {@code [1, anchors, 4 × 8]}.
     */
    private List<Detection> decode(List<Tensor> outputs, int width, int height,
                                   float confidenceThreshold, float iouThreshold) {
        if (outputs.size() != 2 * STRIDES.length) {
            throw new InferenceException("Expected " + 2 * STRIDES.length
                    + " PicoDet outputs (scores and box distributions for strides 8, 16, 32, 64), got "
                    + outputs.size());
        }
        float scaleX = (float) inputSize / width;
        float scaleY = (float) inputSize / height;
        List<Detection> candidates = new ArrayList<>();
        for (int level = 0; level < STRIDES.length; level++) {
            int stride = STRIDES[level];
            int cells = (inputSize + stride - 1) / stride;
            Tensor scoreTensor = outputs.get(level);
            Tensor boxTensor = outputs.get(STRIDES.length + level);
            long[] scoreShape = scoreTensor.shape();
            long[] boxShape = boxTensor.shape();
            if (scoreShape.length != 3 || scoreShape[1] != (long) cells * cells
                    || !Arrays.equals(boxShape, new long[]{1, (long) cells * cells, 4 * BINS})) {
                throw new InferenceException("Unexpected PicoDet outputs for stride " + stride + ": scores "
                        + Arrays.toString(scoreShape) + ", boxes " + Arrays.toString(boxShape));
            }
            int classes = (int) scoreShape[2];
            float[] scores = scoreTensor.toFloats();
            float[] distributions = boxTensor.toFloats();
            for (int anchor = 0; anchor < cells * cells; anchor++) {
                float[] box = null;
                for (int c = 0; c < classes; c++) {
                    float score = scores[anchor * classes + c];
                    if (score <= confidenceThreshold) {
                        continue;
                    }
                    if (box == null) {
                        box = box(distributions, anchor, cells, stride, scaleX, scaleY);
                    }
                    candidates.add(new Detection(clamp(box, width, height), labels.get(c), c, score));
                }
            }
        }
        candidates.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
        return DetectionOps.nms(candidates, iouThreshold, true, maxDetections);
    }

    /** The box of one anchor in original-image pixels: the expected distance of each side, from its bins. */
    private static float[] box(float[] distributions, int anchor, int cells, int stride, float scaleX, float scaleY) {
        float ax = anchor % cells + 0.5f;
        float ay = anchor / cells + 0.5f;
        float[] distance = new float[4];
        for (int side = 0; side < 4; side++) {
            int offset = anchor * 4 * BINS + side * BINS;
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < BINS; i++) {
                max = Math.max(max, distributions[offset + i]);
            }
            float sum = 0f;
            float expected = 0f;
            for (int i = 0; i < BINS; i++) {
                float p = (float) Math.exp(distributions[offset + i] - max);
                sum += p;
                expected += p * i;
            }
            distance[side] = expected / sum;
        }
        return new float[]{
                (ax - distance[0]) * stride / scaleX,
                (ay - distance[1]) * stride / scaleY,
                (ax + distance[2]) * stride / scaleX,
                (ay + distance[3]) * stride / scaleY};
    }

    private static BoundingBox clamp(float[] box, int width, int height) {
        return new BoundingBox(
                Math.max(0f, Math.min(box[0], width)),
                Math.max(0f, Math.min(box[1], height)),
                Math.max(0f, Math.min(box[2], width)),
                Math.max(0f, Math.min(box[3], height)));
    }

    public static class Builder extends AbstractModelBuilder<PicoDetDetector, Builder> {
        private Labels labels;
        private Integer inputSize;
        private float confidenceThreshold = 0.5f;
        private float iouThreshold = 0.6f;
        private int maxDetections = DetectorModels.DEFAULT_MAX_DETECTIONS;

        public Builder() {
            super(S);
        }

        /**
         * Class names, by class index. When not set: the 80 COCO classes for the presets, and for a custom model
         * a {@code labels.txt} or the {@code id2label} of a {@code config.json} next to it, falling back to COCO.
         */
        public Builder labels(Labels labels) {
            this.labels = labels;
            return this;
        }

        /**
         * Side of the square model input, in pixels. A model with a fixed input size uses that size, and a different
         * value fails at {@link #build()}; otherwise defaults to 640. The presets have a fixed 416 × 416 input.
         */
        public Builder inputSize(int inputSize) {
            this.inputSize = inputSize;
            return this;
        }

        /** Detections scoring at or below this are dropped. Defaults to 0.5, as in PaddleDetection. */
        public Builder confidenceThreshold(float confidenceThreshold) {
            this.confidenceThreshold = confidenceThreshold;
            return this;
        }

        /** IoU above which a less confident box of the same class is removed. Defaults to 0.6, as in PaddleDetection. */
        public Builder iouThreshold(float iouThreshold) {
            this.iouThreshold = iouThreshold;
            return this;
        }

        /** The most detections returned per image, the most confident first. Defaults to 300. */
        public Builder maxDetections(int maxDetections) {
            this.maxDetections = DetectorModels.maxDetections(maxDetections);
            return this;
        }

        public PicoDetDetector build() {
            String id = resolvedModelId();
            return load(List.of("model.onnx"), files -> {
                InferenceSession session = files.session("model.onnx");
                String inputName = session.inputNames().iterator().next();
                int size = DetectorModels.inputSize(inputSize,
                        DetectorModels.fixedSize(session.inputShape(inputName)), id);
                Labels resolved = DetectorModels.labels(labels, files.directory(), PRESETS.contains(id));
                return new PicoDetDetector(session, inputName, resolved, size, confidenceThreshold, iouThreshold,
                        maxDetections);
            });
        }
    }
}
