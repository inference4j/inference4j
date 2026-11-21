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
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Transformer object detector for the DETR family: D-FINE and RT-DETRv2, both <strong>Apache-2.0</strong>.
 *
 * <p>Unlike the YOLO wrappers, whose Ultralytics models are AGPL-3.0, these models can be used in proprietary
 * software. They are NMS-free: the model returns 300 object queries, each with class scores and a box, and the
 * detections are simply the highest-scoring query and class pairs.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * try (var detector = DetrDetector.dfineSmall().build()) {   // D-FINE small, 41 MB (the default)
 *     List<Detection> detections = detector.detect(Path.of("street.jpg"));
 * }
 * DetrDetector.dfineNano()    // D-FINE nano, 15 MB, fastest
 * DetrDetector.dfineMedium()  // D-FINE medium, 79 MB, more accurate
 * DetrDetector.dfineLarge()   // D-FINE large, 125 MB, the most accurate
 * DetrDetector.rtDetrV2()     // RT-DETRv2 ResNet-18, 81 MB
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
public class DetrDetector implements ObjectDetector {

    private static final String DFINE_SMALL = "inference4j/dfine-small-coco";
    private static final String DFINE_NANO = "inference4j/dfine-nano-coco";
    private static final String DFINE_MEDIUM = "inference4j/dfine-medium-coco";
    private static final String DFINE_LARGE = "inference4j/dfine-large-coco";
    private static final String RTDETR_V2 = "inference4j/rtdetr-v2-r18vd";
    private static final Set<String> PRESETS = Set.of(DFINE_SMALL, DFINE_NANO, DFINE_MEDIUM, DFINE_LARGE, RTDETR_V2);

    private final InferenceSession session;
    private final String inputName;
    private final Labels labels;
    private final int inputSize;
    private final float defaultThreshold;
    private final int maxDetections;

    DetrDetector(Loaded loaded) {
        this.session = loaded.session();
        this.inputName = loaded.inputName();
        this.labels = loaded.labels();
        this.inputSize = loaded.inputSize();
        this.defaultThreshold = loaded.confidenceThreshold();
        this.maxDetections = loaded.maxDetections();
    }

    /** What a builder resolved: the open session and the settings the detector runs with. */
    record Loaded(InferenceSession session, String inputName, Labels labels, int inputSize,
                  float confidenceThreshold, int maxDetections) {
    }

    /** D-FINE small (41 MB), the default. */
    public static Builder dfineSmall() {
        return new Builder().modelId(DFINE_SMALL);
    }

    /** D-FINE nano (15 MB): the fastest, slightly less accurate. */
    public static Builder dfineNano() {
        return new Builder().modelId(DFINE_NANO);
    }

    /** D-FINE medium (79 MB): more accurate than small (52.3 against 48.5 COCO AP, the authors' figures), slower. */
    public static Builder dfineMedium() {
        return new Builder().modelId(DFINE_MEDIUM);
    }

    /** D-FINE large (125 MB): the most accurate preset (54.0 COCO AP, the authors' figure), and the slowest. */
    public static Builder dfineLarge() {
        return new Builder().modelId(DFINE_LARGE);
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
        float[] input = DetectionOps.toChw(resized, inputSize, inputSize);   // equals HuggingFace's rescale by 1/255
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
        // HuggingFace's post_process_object_detection for RT-DETR with focal loss
        return DetectionOps.decodeQueries(logits.toFloats(), boxes.toFloats(), (int) shape[1], (int) shape[2],
                confidenceThreshold, width, height, labels, DetectionOps.QueryDecoding.TOP_PAIRS,
                maxDetections);
    }

    /** Detects objects with a confidence threshold other than the configured one. */
    @Override
    public List<Detection> detect(BufferedImage image, float confidenceThreshold) {
        return detect(image, confidenceThreshold, 0f);
    }

    @Override
    public void close() {
        session.close();
    }

    public static class Builder {
        ModelSource modelSource;
        String modelId = DFINE_SMALL;
        SessionConfigurer sessionConfigurer;
        Labels labels;
        Integer inputSize;
        float confidenceThreshold = 0.5f;
        int maxDetections = DetectorModels.DEFAULT_MAX_DETECTIONS;

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
         * a {@code labels.txt} or the {@code id2label} of its {@code config.json}, falling back to COCO.
         */
        public Builder labels(Labels labels) {
            this.labels = labels;
            return this;
        }

        /**
         * Side of the square model input, in pixels. A model with a fixed input size uses that size, and a different
         * value fails at {@link #build()}; otherwise defaults to 640, the size the presets were trained at.
         */
        public Builder inputSize(int inputSize) {
            this.inputSize = inputSize;
            return this;
        }

        /** Detections scoring at or below this are dropped. Defaults to 0.5, as in HuggingFace. */
        public Builder confidenceThreshold(float confidenceThreshold) {
            this.confidenceThreshold = confidenceThreshold;
            return this;
        }

        /** The most detections returned per image, the most confident first. Defaults to 300. */
        public Builder maxDetections(int maxDetections) {
            this.maxDetections = DetectorModels.maxDetections(maxDetections);
            return this;
        }

        public DetrDetector build() {
            return new DetrDetector(load());
        }

        Loaded load() {
            ModelSource source = modelSource != null ? modelSource : HuggingFaceModelSource.defaultInstance();
            boolean preset = PRESETS.contains(modelId);
            Path dir = preset ? source.resolve(modelId, List.of("model.onnx")) : resolveCustom(source);
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
                return new Loaded(session, inputName, DetectorModels.labels(labels, dir, preset), size,
                        confidenceThreshold, maxDetections);
            } catch (RuntimeException e) {
                session.close();
                throw e;
            }
        }

        /** A custom model's directory, with its {@code config.json} when the repository has one. */
        private Path resolveCustom(ModelSource source) {
            try {
                return source.resolve(modelId, List.of("model.onnx", "config.json"));
            } catch (ModelSourceException e) {
                return source.resolve(modelId, List.of("model.onnx"));
            }
        }
    }
}
