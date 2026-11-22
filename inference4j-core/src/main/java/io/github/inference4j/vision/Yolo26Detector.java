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

import io.github.inference4j.AbstractInferenceTask;
import io.github.inference4j.PreprocessResult;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.model.ResolvedModel;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.preprocessing.image.Labels;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * YOLO26 object detector.
 *
 * <p>This wrapper supports the
 * <a href="https://huggingface.co/onnx-community/yolo26x-ONNX">YOLO26</a> NMS-free
 * architecture. Unlike YOLOv8, YOLO26 outputs 300 deduplicated proposals directly —
 * no non-maximum suppression is needed.
 *
 * <p><strong>Not compatible with YOLOv8/YOLO11</strong> (different output layout with
 * NMS-based post-processing).
 *
 * <h2>Tested model</h2>
 * <p>Tested against
 * <a href="https://huggingface.co/onnx-community/yolo26x-ONNX">yolo26x-ONNX</a>.
 * That model outputs two tensors:
 * <ul>
 *   <li>{@code logits}: shape {@code [1, 300, 80]} — raw class scores (sigmoid needed)</li>
 *   <li>{@code pred_boxes}: shape {@code [1, 300, 4]} — normalized {@code [cx, cy, w, h]} in 0–1</li>
 * </ul>
 *
 * <h2>Output identification</h2>
 * <p>Output tensors are identified by shape rather than name — the tensor whose last
 * dimension is 4 is treated as boxes, the other as logits. This is robust across
 * different ONNX exporters.
 *
 * <h2>Preprocessing</h2>
 * <ul>
 *   <li>Resize to {@code inputSize x inputSize} via bilinear interpolation (aspect ratio NOT preserved)</li>
 *   <li>Normalization: {@code pixel / 255} only (no ImageNet mean/std, no letterbox padding)</li>
 *   <li>Input layout: NCHW {@code [1, 3, inputSize, inputSize]}</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (Yolo26Detector detector = Yolo26Detector.builder().build()) {
 *     List<Detection> detections = detector.detect(Path.of("street.jpg"));
 *     for (Detection d : detections) {
 *         System.out.printf("%s (%.2f) at [%.0f, %.0f, %.0f, %.0f]%n",
 *             d.label(), d.confidence(),
 *             d.box().x1(), d.box().y1(), d.box().x2(), d.box().y2());
 *     }
 * }
 * }</pre>
 *
 * <h2>Custom configuration</h2>
 * <pre>{@code
 * try (Yolo26Detector detector = Yolo26Detector.builder()
 *         .modelId("my-org/my-yolo26")
 *         .modelSource(new LocalModelSource(modelsDir))   // resolves modelsDir/<modelId>
 *         .sessionOptions(opts -> opts.addCUDA(0))
 *         .labels(Labels.fromFile(Path.of("custom-labels.txt")))
 *         .confidenceThreshold(0.5f)
 *         .build()) {
 *     List<Detection> detections = detector.detect(image);
 * }
 * }</pre>
 *
 * <h2>License</h2>
 * <p>The wrapper code is Apache-2.0. The default weights, {@code inference4j/yolo26n}
 * (mirrored from <a href="https://huggingface.co/onnx-community/yolo26n-ONNX">onnx-community/yolo26n-ONNX</a>,
 * an <a href="https://www.ultralytics.com/license">Ultralytics</a> model), are licensed under
 * AGPL-3.0. {@link Builder#build()} logs a warning when it loads these default weights; it
 * does not warn for other model ids. {@link DetrDetector} (D-FINE / RT-DETRv2 weights,
 * Apache-2.0) is the permissive alternative. This note is not legal advice; check the
 * license of the weights you deploy.
 *
 * @see Detection
 * @see BoundingBox
 * @see YoloV8Detector
 * @see DetrDetector
 */
public class Yolo26Detector
        extends AbstractInferenceTask<BufferedImage, List<Detection>>
        implements ObjectDetector {

    private static final Logger logger = LoggerFactory.getLogger(Yolo26Detector.class);

    private static final String DEFAULT_MODEL_ID = "inference4j/yolo26n";

    private final Labels labels;
    private final String inputName;
    private final int inputSize;
    private final float defaultConfidenceThreshold;
    private final int maxDetections;

    private Yolo26Detector(InferenceSession session, Labels labels, String inputName,
                           int inputSize, float defaultConfidenceThreshold, int maxDetections) {
        super(session,
                createPreprocessor(inputName, inputSize),
                ctx -> decodeDetections(ctx.outputs(), labels, defaultConfidenceThreshold, maxDetections,
                        ctx.input().getWidth(), ctx.input().getHeight()));
        this.maxDetections = maxDetections;
        this.labels = labels;
        this.inputName = inputName;
        this.inputSize = inputSize;
        this.defaultConfidenceThreshold = defaultConfidenceThreshold;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<Detection> detect(BufferedImage image) {
        return run(image);
    }

    /**
     * Detects objects in the given image.
     *
     * <p>The {@code iouThreshold} parameter is accepted for interface compatibility but
     * is ignored — YOLO26 uses an NMS-free architecture where the model outputs
     * deduplicated proposals directly.
     */
    @Override
    public List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold) {
        PreprocessResult result = preprocessor.process(image);
        Map<String, Tensor> outputs = session.run(result.tensors());
        return decodeDetections(outputs, labels, confidenceThreshold, maxDetections,
                image.getWidth(), image.getHeight());
    }

    /** Detects objects with a confidence threshold other than the configured one. */
    @Override
    public List<Detection> detect(BufferedImage image, float confidenceThreshold) {
        return detect(image, confidenceThreshold, 0f);
    }

    // --- Preprocessing ---

    private static io.github.inference4j.processing.Preprocessor<BufferedImage, PreprocessResult> createPreprocessor(
            String inputName, int inputSize) {
        return image -> {
            BufferedImage resized = resize(image, inputSize);
            Tensor tensor = imageToTensor(resized);
            return PreprocessResult.of(Map.of(inputName, tensor));
        };
    }

    static BufferedImage resize(BufferedImage image, int targetSize) {
        BufferedImage result = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = result.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(image, 0, 0, targetSize, targetSize, null);
        g.dispose();
        return result;
    }

    private static Tensor imageToTensor(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        return Tensor.fromFloats(DetectionOps.toChw(DetectionOps.pixels(image), w, h), new long[]{1, 3, h, w});
    }

    // --- Post-processing ---

    private static List<Detection> decodeDetections(Map<String, Tensor> outputs,
                                                     Labels labels, float confThreshold, int maxDetections,
                                                     int origWidth, int origHeight) {
        float[] logitsData = null;
        long[] logitsShape = null;
        float[] boxesData = null;
        long[] boxesShape = null;

        for (Tensor tensor : outputs.values()) {
            long[] shape = tensor.shape();
            if (shape.length >= 2 && shape[shape.length - 1] == 4) {
                boxesData = tensor.toFloats();
                boxesShape = shape;
            } else {
                logitsData = tensor.toFloats();
                logitsShape = shape;
            }
        }

        if (logitsData == null || boxesData == null) {
            throw new InferenceException(
                    "Expected two output tensors (logits and boxes), got: " + outputs.size());
        }

        return postProcess(logitsData, logitsShape, boxesData, boxesShape,
                labels, confThreshold, maxDetections, origWidth, origHeight);
    }

    static List<Detection> postProcess(float[] logitsData, long[] logitsShape,
                                       float[] boxesData, long[] boxesShape,
                                       Labels labels, float confThreshold,
                                       int origWidth, int origHeight) {
        return postProcess(logitsData, logitsShape, boxesData, boxesShape, labels, confThreshold,
                Integer.MAX_VALUE, origWidth, origHeight);
    }

    static List<Detection> postProcess(float[] logitsData, long[] logitsShape,
                                       float[] boxesData, long[] boxesShape,
                                       Labels labels, float confThreshold, int maxDetections,
                                       int origWidth, int origHeight) {
        return DetectionOps.decodeQueries(logitsData, boxesData, (int) logitsShape[1], (int) logitsShape[2],
                confThreshold, origWidth, origHeight, labels, DetectionOps.QueryDecoding.BEST_CLASS_PER_QUERY,
                maxDetections);
    }

    // --- Utilities ---

    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private Labels labels;
        private String inputName;
        private Integer inputSize;
        private int maxDetections = DetectorModels.DEFAULT_MAX_DETECTIONS;
        private float confidenceThreshold = 0.5f;

        Builder session(InferenceSession session) {
            this.session = session;
            return this;
        }

        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * Class names by class index. When not set: a {@code labels.txt} or the {@code id2label} of a
         * {@code config.json} next to a custom model, otherwise the 80 COCO classes.
         */
        public Builder labels(Labels labels) {
            this.labels = labels;
            return this;
        }

        /** The most detections returned per image, the most confident first. Defaults to 300. */
        public Builder maxDetections(int maxDetections) {
            this.maxDetections = DetectorModels.maxDetections(maxDetections);
            return this;
        }

        public Builder inputName(String inputName) {
            this.inputName = inputName;
            return this;
        }

        /**
         * Side of the square model input, in pixels. A model with a fixed input size uses that size, and a different
         * value fails at {@link #build()}; otherwise defaults to 640.
         */
        public Builder inputSize(int inputSize) {
            this.inputSize = inputSize;
            return this;
        }

        public Builder confidenceThreshold(float confidenceThreshold) {
            this.confidenceThreshold = confidenceThreshold;
            return this;
        }

        /**
         * Sets the IoU threshold. This parameter is accepted for API consistency but
         * has no effect — YOLO26 is NMS-free.
         */
        public Builder iouThreshold(float iouThreshold) {
            // Accepted but ignored — YOLO26 is NMS-free
            return this;
        }

        public Yolo26Detector build() {
            String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
            Path dir = null;
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                if (DEFAULT_MODEL_ID.equals(id)) {
                    logger.warn("Yolo26Detector is using the default weights '{}', which are licensed "
                            + "under AGPL-3.0 (Ultralytics). DetrDetector is an Apache-2.0 alternative.", id);
                }
                try (ResolvedModel model = source.open(id, List.of("model.onnx"))) {
                    dir = model.directory();
                    loadFromDirectory(model);
                }
            }
            try {
                if (inputName == null) {
                    inputName = session.inputNames().iterator().next();
                }
                int fixedSize = dir != null ? DetectorModels.fixedSize(session.inputShape(inputName)) : -1;
                int size = DetectorModels.inputSize(inputSize, fixedSize, id);
                Labels resolved = DetectorModels.labels(labels, dir, DEFAULT_MODEL_ID.equals(id));
                return new Yolo26Detector(session, resolved, inputName, size, confidenceThreshold, maxDetections);
            } catch (RuntimeException e) {
                if (dir != null) {
                    session.close();
                }
                throw e;
            }
        }

        private void loadFromDirectory(ResolvedModel model) {
            Path dir = model.directory();
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(model, "model.onnx", sessionConfigurer)
                    : InferenceSession.create(model, "model.onnx");
        }
    }
}
