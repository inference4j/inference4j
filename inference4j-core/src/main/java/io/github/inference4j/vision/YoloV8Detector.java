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
import io.github.inference4j.InferenceSession;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.Tensor;
import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.model.AbstractModelBuilder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * YOLOv8 object detector.
 *
 * <p>This wrapper supports the
 * <a href="https://docs.ultralytics.com/models/yolov8/">YOLOv8</a> output layout
 * ({@code [1, 4+numClasses, numCandidates]}). It is also compatible with
 * <a href="https://docs.ultralytics.com/models/yolo11/">YOLO11</a>, which uses
 * the same output topology.
 *
 * <p><strong>Not compatible with YOLOv5</strong> (different layout with separate
 * objectness score) or <strong>YOLO26</strong> (NMS-free architecture with
 * different output format).
 *
 * <h2>Tested model</h2>
 * <p>Tested against
 * <a href="https://huggingface.co/Kalray/yolov8">YOLOv8n</a> (Ultralytics architecture).
 * That model outputs shape {@code [1, 84, 8400]} — 4 box coordinates
 * ({@code cx, cy, w, h}) plus 80 COCO class scores for 8400 candidate detections.
 * Class scores have sigmoid already applied (no additional activation needed).
 *
 * <h2>Compatibility</h2>
 * <table>
 *   <tr><th>Version</th><th>Compatible</th><th>Output layout</th></tr>
 *   <tr><td>YOLOv8</td><td>Yes</td><td>{@code [1, 4+C, N]} — no objectness score</td></tr>
 *   <tr><td>YOLO11</td><td>Yes</td><td>{@code [1, 4+C, N]} — same as v8</td></tr>
 *   <tr><td>YOLOv5</td><td>No</td><td>{@code [1, N, 5+C]} — has objectness column</td></tr>
 *   <tr><td>YOLO26</td><td>No</td><td>NMS-free, different output</td></tr>
 * </table>
 *
 * <h2>Preprocessing</h2>
 * <ul>
 *   <li>Letterbox resize: preserves aspect ratio, pads with gray (114/255)</li>
 *   <li>Normalization: {@code pixel / 255} only (no ImageNet mean/std)</li>
 *   <li>Input layout: NCHW {@code [1, 3, inputSize, inputSize]}</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (YoloV8Detector detector = YoloV8Detector.builder().build()) {
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
 * try (YoloV8Detector detector = YoloV8Detector.builder()
 *         .modelId("my-org/my-yolov8")
 *         .modelSource(new LocalModelSource(modelsDir))   // resolves modelsDir/<modelId>
 *         .sessionOptions(opts -> opts.addCUDA(0))
 *         .labels(Labels.fromFile(Path.of("custom-labels.txt")))
 *         .confidenceThreshold(0.5f)
 *         .iouThreshold(0.4f)
 *         .build()) {
 *     List<Detection> detections = detector.detect(image);
 * }
 * }</pre>
 *
 * <h2>License</h2>
 * <p>The wrapper code is Apache-2.0. The default weights, {@code inference4j/yolov8n}
 * (mirrored from <a href="https://huggingface.co/Kalray/yolov8">Kalray/yolov8</a>, an
 * <a href="https://www.ultralytics.com/license">Ultralytics</a> model), are licensed under
 * AGPL-3.0. {@link Builder#build()} logs a warning when it loads these default weights; it
 * does not warn for other model ids. {@link DetrDetector} (D-FINE / RT-DETRv2 weights,
 * Apache-2.0) is the permissive alternative. This note is not legal advice; check the
 * license of the weights you deploy.
 *
 * @see Detection
 * @see BoundingBox
 * @see Yolo26Detector
 * @see DetrDetector
 */
public class YoloV8Detector
        extends AbstractInferenceTask<BufferedImage, List<Detection>>
        implements ObjectDetector {

    private static final Logger logger = LoggerFactory.getLogger(YoloV8Detector.class);

    private static final String DEFAULT_MODEL_ID = "inference4j/yolov8n";
    private static final Color PAD_COLOR = new Color(114, 114, 114);

    private final Labels labels;
    private final String inputName;
    private final int inputSize;
    private final float defaultConfidenceThreshold;
    private final float defaultIouThreshold;
    private final int maxDetections;

    private YoloV8Detector(InferenceSession session, Labels labels, String inputName,
                           int inputSize, float defaultConfidenceThreshold, float defaultIouThreshold,
                           int maxDetections) {
        super(session,
                createPreprocessor(inputName, inputSize),
                ctx -> decodeDetections(ctx.outputs(), labels,
                        defaultConfidenceThreshold, defaultIouThreshold, maxDetections,
                        inputSize, ctx.input().getWidth(), ctx.input().getHeight()));
        this.maxDetections = maxDetections;
        this.labels = labels;
        this.inputName = inputName;
        this.inputSize = inputSize;
        this.defaultConfidenceThreshold = defaultConfidenceThreshold;
        this.defaultIouThreshold = defaultIouThreshold;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<Detection> detect(BufferedImage image) {
        return run(image);
    }

    @Override
    public List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold) {
        PreprocessResult result = preprocessor.process(image);
        Map<String, Tensor> outputs = session.run(result.tensors());
        return decodeDetections(outputs, labels, confidenceThreshold, iouThreshold, maxDetections,
                inputSize, image.getWidth(), image.getHeight());
    }

    /** Detects objects with a confidence threshold other than the configured one, and the configured IoU threshold. */
    @Override
    public List<Detection> detect(BufferedImage image, float confidenceThreshold) {
        return detect(image, confidenceThreshold, defaultIouThreshold);
    }

    // --- Preprocessing ---

    private static io.github.inference4j.processing.Preprocessor<BufferedImage, PreprocessResult> createPreprocessor(
            String inputName, int inputSize) {
        return image -> {
            LetterboxResult lb = letterbox(image, inputSize);
            Tensor tensor = imageToTensor(lb.image);
            return PreprocessResult.of(Map.of(inputName, tensor));
        };
    }

    record LetterboxResult(BufferedImage image, float scale, float padX, float padY) {
    }

    static LetterboxResult letterbox(BufferedImage image, int targetSize) {
        DetectionOps.Letterbox letterbox = DetectionOps.Letterbox.centered(image.getWidth(), image.getHeight(),
                targetSize);
        return new LetterboxResult(letterbox.apply(image, PAD_COLOR), letterbox.scale(), letterbox.padX(),
                letterbox.padY());
    }

    private static Tensor imageToTensor(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        return Tensor.fromFloats(DetectionOps.toChw(DetectionOps.pixels(image), w, h), new long[]{1, 3, h, w});
    }

    // --- Post-processing ---

    private static List<Detection> decodeDetections(Map<String, Tensor> outputs,
                                                     Labels labels,
                                                     float confThreshold, float iouThreshold,
                                                     int maxDetections, int inputSize,
                                                     int origWidth, int origHeight) {
        Tensor outputTensor = outputs.values().iterator().next();

        DetectionOps.Letterbox letterbox = DetectionOps.Letterbox.centered(origWidth, origHeight, inputSize);
        float scale = letterbox.scale();
        float padX = letterbox.padX();
        float padY = letterbox.padY();

        return postProcess(outputTensor.toFloats(), outputTensor.shape(),
                labels, confThreshold, iouThreshold, maxDetections,
                scale, padX, padY, origWidth, origHeight);
    }

    static List<Detection> postProcess(float[] rawOutput, long[] outputShape,
                                       Labels labels,
                                       float confThreshold, float iouThreshold,
                                       float scale, float padX, float padY,
                                       int origWidth, int origHeight) {
        return postProcess(rawOutput, outputShape, labels, confThreshold, iouThreshold, Integer.MAX_VALUE,
                scale, padX, padY, origWidth, origHeight);
    }

    static List<Detection> postProcess(float[] rawOutput, long[] outputShape,
                                       Labels labels,
                                       float confThreshold, float iouThreshold, int maxDetections,
                                       float scale, float padX, float padY,
                                       int origWidth, int origHeight) {
        int numOutputs = (int) outputShape[1];
        int numCandidates = (int) outputShape[2];
        int numClasses = numOutputs - 4;

        DetectionOps.Letterbox letterbox = new DetectionOps.Letterbox(scale, padX, padY, 0, 0, 0);
        List<Detection> candidates = new ArrayList<>();
        for (int c = 0; c < numCandidates; c++) {
            int bestClass = -1;
            float bestScore = -1f;
            for (int cls = 0; cls < numClasses; cls++) {
                float score = rawOutput[(4 + cls) * numCandidates + c];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = cls;
                }
            }

            if (bestScore < confThreshold) {
                continue;
            }

            float[] xyxy = MathOps.cxcywh2xyxy(new float[]{
                    rawOutput[c], rawOutput[numCandidates + c],
                    rawOutput[2 * numCandidates + c], rawOutput[3 * numCandidates + c]});
            candidates.add(new Detection(letterbox.toOriginal(xyxy, origWidth, origHeight),
                    labels.get(bestClass), bestClass, bestScore));
        }

        // Class-agnostic, as before: one box per object whatever its class
        return DetectionOps.nms(candidates, iouThreshold, false, maxDetections);
    }

    // --- Utilities ---

    public static class Builder extends AbstractModelBuilder<YoloV8Detector, Builder> {
        private Labels labels;
        private String inputName;
        private Integer inputSize;
        private int maxDetections = DetectorModels.DEFAULT_MAX_DETECTIONS;
        private float confidenceThreshold = 0.25f;
        private float iouThreshold = 0.45f;

        public Builder() {
            super(DEFAULT_MODEL_ID);
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

        public Builder iouThreshold(float iouThreshold) {
            this.iouThreshold = iouThreshold;
            return this;
        }

        @Override
        public YoloV8Detector build() {
            String id = resolvedModelId();
            boolean preset = DEFAULT_MODEL_ID.equals(id);
            if (preset) {
                DetectorModels.warnAgpl(logger, "YoloV8Detector", id);
            }
            return load(List.of("model.onnx"), files -> {
                InferenceSession session = files.session("model.onnx");
                String input = inputName != null ? inputName : session.inputNames().iterator().next();
                int size = DetectorModels.inputSize(inputSize, DetectorModels.fixedSize(session.inputShape(input)), id);
                Labels resolved = DetectorModels.labels(labels, files.directory(), preset);
                return new YoloV8Detector(session, resolved, input, size, confidenceThreshold, iouThreshold,
                        maxDetections);
            });
        }
    }
}
