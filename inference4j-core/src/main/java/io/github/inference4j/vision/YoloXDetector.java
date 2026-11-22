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
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.session.SessionConfigurer;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Object detector for <a href="https://github.com/Megvii-BaseDetection/YOLOX">YOLOX</a> (Megvii), an anchor-free
 * CNN detector. Code and weights are Apache-2.0.
 *
 * <pre>{@code
 * try (ObjectDetector detector = YoloXDetector.nano().build()) {
 *     List<Detection> detections = detector.detect(Path.of("street.jpg"));
 * }
 * }</pre>
 *
 * <p>Presets, trained on COCO by the YOLOX authors and published unchanged from their release:
 * <ul>
 *   <li>{@link #nano()} (default): 416 × 416 input, 25.8 COCO AP, 3.5 MB;</li>
 *   <li>{@link #tiny()}: 416 × 416, 32.8 AP, 19 MB;</li>
 *   <li>{@link #s()}: 640 × 640, 40.5 AP, 34 MB.</li>
 * </ul>
 *
 * <p>Matches the official ONNX Runtime demo: the image is scaled to fit the input, keeping its aspect ratio,
 * with OpenCV's bilinear resize, placed at the top left and padded with gray (114). The input is BGR with raw
 * 0–255 values. Each output row is decoded on its grid cell and stride (8, 16, 32), scored as objectness × class
 * score, and overlapping boxes are removed with class-agnostic non-maximum suppression.
 *
 * <p>Labels default to {@link Labels#coco()}, the same as the other detectors.
 *
 * @see ObjectDetector
 * @see Detection
 */
public class YoloXDetector implements ObjectDetector {

    private static final String NANO = "inference4j/yolox-nano";
    private static final String TINY = "inference4j/yolox-tiny";
    private static final String S = "inference4j/yolox-s";
    private static final Set<String> PRESETS = Set.of(NANO, TINY, S);
    private static final int[] STRIDES = {8, 16, 32};
    private static final float PAD_VALUE = 114f;

    private final InferenceSession session;
    private final String inputName;
    private final Labels labels;
    private final int inputSize;
    private final float defaultConfidenceThreshold;
    private final float defaultIouThreshold;
    private final int maxDetections;

    private YoloXDetector(InferenceSession session, String inputName, Labels labels, int inputSize,
                          float confidenceThreshold, float iouThreshold, int maxDetections) {
        this.session = session;
        this.inputName = inputName;
        this.labels = labels;
        this.inputSize = inputSize;
        this.defaultConfidenceThreshold = confidenceThreshold;
        this.defaultIouThreshold = iouThreshold;
        this.maxDetections = maxDetections;
    }

    /** YOLOX-Nano (3.5 MB, 416 × 416): the fastest and smallest, 25.8 COCO AP (the authors' figure). The default. */
    public static Builder nano() {
        return new Builder().modelId(NANO);
    }

    /** YOLOX-Tiny (19 MB, 416 × 416): 32.8 COCO AP (the authors' figure). */
    public static Builder tiny() {
        return new Builder().modelId(TINY);
    }

    /** YOLOX-S (34 MB, 640 × 640): 40.5 COCO AP (the authors' figure), the slowest preset. */
    public static Builder s() {
        return new Builder().modelId(S);
    }

    /** A builder for the default model, {@link #nano()}. */
    public static Builder builder() {
        return nano();
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
        DetectionOps.Letterbox letterbox = DetectionOps.Letterbox.topLeft(width, height, inputSize);
        float[] input = toInput(image, letterbox);
        Map<String, Tensor> outputs = session.run(Map.of(inputName,
                Tensor.fromFloats(input, new long[]{1, 3, inputSize, inputSize})));
        Tensor output = outputs.values().iterator().next();
        return decode(output.toFloats(), output.shape(), letterbox, width, height, confidenceThreshold, iouThreshold);
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

    /** The official demo's {@code demo_postprocess} and {@code multiclass_nms}, on {@code [1, rows, 5 + classes]}. */
    private List<Detection> decode(float[] output, long[] shape, DetectionOps.Letterbox letterbox,
                                   int width, int height, float confidenceThreshold, float iouThreshold) {
        int expectedRows = 0;
        for (int stride : STRIDES) {
            expectedRows += (inputSize / stride) * (inputSize / stride);
        }
        if (shape.length != 3 || shape[1] != expectedRows || shape[2] <= 5) {
            throw new InferenceException("Expected a YOLOX output of shape [1, " + expectedRows
                    + ", 5 + classes] for a " + inputSize + " × " + inputSize + " input, got "
                    + Arrays.toString(shape));
        }
        int columns = (int) shape[2];
        int classes = columns - 5;
        List<Detection> candidates = new ArrayList<>();
        int row = 0;
        for (int stride : STRIDES) {
            int cells = inputSize / stride;
            for (int gy = 0; gy < cells; gy++) {
                for (int gx = 0; gx < cells; gx++, row++) {
                    int base = row * columns;
                    float objectness = output[base + 4];
                    int bestClass = 0;
                    float bestScore = output[base + 5];
                    for (int c = 1; c < classes; c++) {
                        if (output[base + 5 + c] > bestScore) {
                            bestScore = output[base + 5 + c];
                            bestClass = c;
                        }
                    }
                    float score = objectness * bestScore;
                    if (score < confidenceThreshold) {
                        continue;
                    }
                    float cx = (output[base] + gx) * stride;
                    float cy = (output[base + 1] + gy) * stride;
                    float w = (float) Math.exp(output[base + 2]) * stride;
                    float h = (float) Math.exp(output[base + 3]) * stride;
                    float[] box = {cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2};
                    candidates.add(new Detection(letterbox.toOriginal(box, width, height),
                            labels.get(bestClass), bestClass, score));
                }
            }
        }
        candidates.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
        return DetectionOps.nms(candidates, iouThreshold, false, maxDetections);
    }

    public static class Builder {
        private ModelSource modelSource;
        private String modelId = NANO;
        private SessionConfigurer sessionConfigurer;
        private Labels labels;
        private Integer inputSize;
        private float confidenceThreshold = 0.3f;
        private float iouThreshold = 0.45f;
        private int maxDetections = DetectorModels.DEFAULT_MAX_DETECTIONS;

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
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
         * value fails at {@link #build()}; otherwise defaults to 640. The presets have fixed inputs.
         */
        public Builder inputSize(int inputSize) {
            this.inputSize = inputSize;
            return this;
        }

        /** Detections scoring below this (objectness × class score) are dropped. Defaults to 0.3, as in the demo. */
        public Builder confidenceThreshold(float confidenceThreshold) {
            this.confidenceThreshold = confidenceThreshold;
            return this;
        }

        /** IoU above which a less confident box is removed, whatever its class. Defaults to 0.45. */
        public Builder iouThreshold(float iouThreshold) {
            this.iouThreshold = iouThreshold;
            return this;
        }

        /** The most detections returned per image, the most confident first. Defaults to 300. */
        public Builder maxDetections(int maxDetections) {
            this.maxDetections = DetectorModels.maxDetections(maxDetections);
            return this;
        }

        public YoloXDetector build() {
            ModelSource source = modelSource != null ? modelSource : HuggingFaceModelSource.defaultInstance();
            Path dir = source.resolve(modelId, List.of("model.onnx"));
            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }
            InferenceSession session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            try {
                String inputName = session.inputNames().iterator().next();
                int size = DetectorModels.inputSize(inputSize,
                        DetectorModels.fixedSize(session.inputShape(inputName)), modelId);
                Labels resolved = DetectorModels.labels(labels, dir, PRESETS.contains(modelId));
                return new YoloXDetector(session, inputName, resolved, size, confidenceThreshold, iouThreshold,
                        maxDetections);
            } catch (RuntimeException e) {
                session.close();
                throw e;
            }
        }
    }
}
