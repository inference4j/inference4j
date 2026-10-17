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
import java.util.List;
import java.util.Map;

/**
 * Transformer object detector for the RT-DETR family: RT-DETRv2 and D-FINE, both <strong>Apache-2.0</strong>.
 *
 * <p>Unlike the YOLO wrappers, whose Ultralytics models are AGPL-3.0, these models can be used in proprietary
 * software. They are NMS-free: the model returns 300 object queries, each with class scores and a box, and the
 * detections are simply the highest-scoring query and class pairs.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * try (var detector = RtDetrDetector.dfineSmall().build()) {   // D-FINE small, 41 MB (the default)
 *     List<Detection> detections = detector.detect(Path.of("street.jpg"));
 * }
 * RtDetrDetector.dfineNano()    // D-FINE nano, 15 MB, fastest
 * RtDetrDetector.rtDetrV2()     // RT-DETRv2 ResNet-18, 81 MB
 * }</pre>
 *
 * <h2>Processing</h2>
 * <p>Matches HuggingFace {@code RTDetrImageProcessor}: the image is resized to 640 × 640 with Pillow's bilinear
 * filter (no letterboxing) and scaled to [0, 1]. Scores are the sigmoid of the class logits; the 300 best
 * (query, class) pairs over all queries are kept, so one query can yield two classes, and those above the
 * confidence threshold become detections, boxes scaled from normalized {@code cx, cy, w, h} to the image.
 *
 * <p>Labels default to {@link Labels#coco()}, the same as the YOLO wrappers, so detectors can be swapped without
 * label names changing.
 *
 * @see ObjectDetector
 * @see Detection
 */
public class RtDetrDetector implements ObjectDetector {

    private static final String DFINE_SMALL = "inference4j/dfine-small-coco";
    private static final String DFINE_NANO = "inference4j/dfine-nano-coco";
    private static final String RTDETR_V2 = "inference4j/rtdetr-v2-r18vd";
    private static final double RESCALE = 0.00392156862745098;   // HuggingFace's 1/255

    private final InferenceSession session;
    private final String inputName;
    private final Labels labels;
    private final int inputSize;
    private final float defaultThreshold;

    private RtDetrDetector(InferenceSession session, Labels labels, int inputSize, float defaultThreshold) {
        this.session = session;
        this.inputName = session.inputNames().iterator().next();
        this.labels = labels;
        this.inputSize = inputSize;
        this.defaultThreshold = defaultThreshold;
    }

    /** D-FINE small (41 MB), the default. */
    public static Builder dfineSmall() {
        return new Builder().modelId(DFINE_SMALL);
    }

    /** D-FINE nano (15 MB): the fastest, slightly less accurate. */
    public static Builder dfineNano() {
        return new Builder().modelId(DFINE_NANO);
    }

    /** RT-DETRv2 with a ResNet-18 backbone (81 MB). */
    public static Builder rtDetrV2() {
        return new Builder().modelId(RTDETR_V2);
    }

    /** A builder for the default model, {@link #dfineSmall()}. */
    public static Builder builder() {
        return dfineSmall();
    }

    @Override
    public List<Detection> detect(BufferedImage image) {
        return detect(image, defaultThreshold, 0f);
    }

    /**
     * Detects objects with a confidence threshold other than the configured one. The {@code iouThreshold} is
     * ignored: these models need no non-maximum suppression.
     */
    @Override
    public List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold) {
        int width = image.getWidth();
        int height = image.getHeight();
        int[] resized = PilImages.resizeBilinear(PaddleImages.pixels(image), width, height, inputSize, inputSize);
        int plane = inputSize * inputSize;
        float[] input = new float[3 * plane];
        for (int i = 0; i < plane; i++) {
            int rgb = resized[i];
            input[i] = (float) (((rgb >> 16) & 0xFF) * RESCALE);
            input[plane + i] = (float) (((rgb >> 8) & 0xFF) * RESCALE);
            input[2 * plane + i] = (float) ((rgb & 0xFF) * RESCALE);
        }
        Map<String, Tensor> outputs = session.run(Map.of(inputName,
                Tensor.fromFloats(input, new long[]{1, 3, inputSize, inputSize})));

        Tensor logits = null;
        Tensor boxes = null;
        for (Tensor tensor : outputs.values()) {
            long[] shape = tensor.shape();
            if (shape[shape.length - 1] == 4) {
                boxes = tensor;
            } else {
                logits = tensor;
            }
        }
        if (logits == null || boxes == null) {
            throw new InferenceException("Expected logits and box outputs, got " + outputs.keySet());
        }
        long[] shape = logits.shape();
        return decode(logits.toFloats(), boxes.toFloats(), (int) shape[1], (int) shape[2],
                confidenceThreshold, width, height);
    }

    @Override
    public List<Detection> detect(Path imagePath) {
        return detect(PaddleImages.read(imagePath));
    }

    /** Detects objects with a confidence threshold other than the configured one; {@code iouThreshold} is ignored. */
    @Override
    public List<Detection> detect(Path imagePath, float confidenceThreshold, float iouThreshold) {
        return detect(PaddleImages.read(imagePath), confidenceThreshold, iouThreshold);
    }

    @Override
    public void close() {
        session.close();
    }

    /**
     * HuggingFace's {@code post_process_object_detection} for RT-DETR with focal loss: the {@code queries} best
     * (query, class) scores overall, above the threshold, best first.
     */
    private List<Detection> decode(float[] logits, float[] boxes, int queries, int classes,
                                   float threshold, int width, int height) {
        List<float[]> candidates = new ArrayList<>();   // score, query, class
        for (int q = 0; q < queries; q++) {
            for (int c = 0; c < classes; c++) {
                float score = (float) (1.0 / (1.0 + Math.exp(-logits[q * classes + c])));
                if (score > threshold) {
                    candidates.add(new float[]{score, q, c});
                }
            }
        }
        // Taking the best `queries` above the threshold equals HuggingFace's top-k followed by the threshold
        candidates.sort((a, b) -> Float.compare(b[0], a[0]));
        List<Detection> detections = new ArrayList<>();
        for (int i = 0; i < Math.min(queries, candidates.size()); i++) {
            float[] candidate = candidates.get(i);
            int q = (int) candidate[1];
            int c = (int) candidate[2];
            float cx = boxes[q * 4];
            float cy = boxes[q * 4 + 1];
            float w = boxes[q * 4 + 2];
            float h = boxes[q * 4 + 3];
            float x1 = clamp((cx - 0.5f * w) * width, width);
            float y1 = clamp((cy - 0.5f * h) * height, height);
            float x2 = clamp((cx + 0.5f * w) * width, width);
            float y2 = clamp((cy + 0.5f * h) * height, height);
            detections.add(new Detection(new BoundingBox(x1, y1, x2, y2), labels.get(c), c, candidate[0]));
        }
        return detections;
    }

    private static float clamp(float value, int max) {
        return Math.max(0, Math.min(value, max));
    }

    public static class Builder {
        private ModelSource modelSource;
        private String modelId = DFINE_SMALL;
        private SessionConfigurer sessionConfigurer;
        private Labels labels = Labels.coco();
        private int inputSize = 640;
        private float confidenceThreshold = 0.5f;

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

        /** Class names, by class index. Defaults to the 80 COCO classes. */
        public Builder labels(Labels labels) {
            this.labels = labels;
            return this;
        }

        /** Side of the square model input, in pixels. Defaults to 640, the size the models were trained at. */
        public Builder inputSize(int inputSize) {
            this.inputSize = inputSize;
            return this;
        }

        /** Detections scoring at or below this are dropped. Defaults to 0.5, as in HuggingFace. */
        public Builder confidenceThreshold(float confidenceThreshold) {
            this.confidenceThreshold = confidenceThreshold;
            return this;
        }

        public RtDetrDetector build() {
            ModelSource source = modelSource != null ? modelSource : HuggingFaceModelSource.defaultInstance();
            Path dir = source.resolve(modelId, List.of("model.onnx"));
            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }
            InferenceSession session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            return new RtDetrDetector(session, labels, inputSize, confidenceThreshold);
        }
    }
}
