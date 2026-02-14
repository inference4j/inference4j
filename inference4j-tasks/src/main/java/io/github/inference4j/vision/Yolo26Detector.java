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

import io.github.inference4j.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.MathOps;
import io.github.inference4j.ModelSource;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.image.ImageLayout;
import io.github.inference4j.image.Labels;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
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
 *         .session(InferenceSession.create(modelPath))
 *         .labels(Labels.fromFile(Path.of("custom-labels.txt")))
 *         .inputSize(640)
 *         .confidenceThreshold(0.5f)
 *         .build()) {
 *     List<Detection> detections = detector.detect(image);
 * }
 * }</pre>
 *
 * @see Detection
 * @see BoundingBox
 * @see YoloV8Detector
 */
public class Yolo26Detector implements ObjectDetector {

    private static final String DEFAULT_MODEL_ID = "inference4j/yolo26n";

    private final InferenceSession session;
    private final Labels labels;
    private final String inputName;
    private final int inputSize;
    private final float defaultConfidenceThreshold;

    private Yolo26Detector(InferenceSession session, Labels labels, String inputName,
                           int inputSize, float defaultConfidenceThreshold) {
        this.session = session;
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
        return detect(image, defaultConfidenceThreshold, 0f);
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
        int origW = image.getWidth();
        int origH = image.getHeight();

        BufferedImage resized = resize(image, inputSize);
        Tensor inputTensor = imageToTensor(resized);

        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put(inputName, inputTensor);

        Map<String, Tensor> outputs = session.run(inputs);

        // Identify outputs by shape: last dim 4 = boxes, other = logits
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
            throw new InferenceException("Expected two output tensors (logits and boxes), got: " + outputs.size());
        }

        return postProcess(logitsData, logitsShape, boxesData, boxesShape,
                labels, confidenceThreshold, origW, origH);
    }

    @Override
    public List<Detection> detect(Path imagePath) {
        return detect(imagePath, defaultConfidenceThreshold, 0f);
    }

    /**
     * Detects objects in the given image file.
     *
     * <p>The {@code iouThreshold} parameter is accepted for interface compatibility but
     * is ignored — YOLO26 uses an NMS-free architecture.
     */
    @Override
    public List<Detection> detect(Path imagePath, float confidenceThreshold, float iouThreshold) {
        return detect(loadImage(imagePath), confidenceThreshold, iouThreshold);
    }

    @Override
    public void close() {
        session.close();
    }

    // --- Preprocessing ---

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
        float[] data = new float[3 * h * w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = image.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255f;
                float g = ((rgb >> 8) & 0xFF) / 255f;
                float b = (rgb & 0xFF) / 255f;

                // NCHW layout, no mean/std normalization (just /255)
                data[0 * h * w + y * w + x] = r;
                data[1 * h * w + y * w + x] = g;
                data[2 * h * w + y * w + x] = b;
            }
        }

        return Tensor.fromFloats(data, new long[]{1, 3, h, w});
    }

    // --- Post-processing ---

    static List<Detection> postProcess(float[] logitsData, long[] logitsShape,
                                       float[] boxesData, long[] boxesShape,
                                       Labels labels, float confThreshold,
                                       int origWidth, int origHeight) {
        int numProposals = (int) logitsShape[1];
        int numClasses = (int) logitsShape[2];

        List<Detection> detections = new ArrayList<>();

        for (int p = 0; p < numProposals; p++) {
            int bestClass = -1;
            float bestScore = -1f;
            for (int cls = 0; cls < numClasses; cls++) {
                float logit = logitsData[p * numClasses + cls];
                float score = (float) (1.0 / (1.0 + Math.exp(-logit)));
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = cls;
                }
            }

            if (bestScore < confThreshold) {
                continue;
            }

            float cx = boxesData[p * 4];
            float cy = boxesData[p * 4 + 1];
            float bw = boxesData[p * 4 + 2];
            float bh = boxesData[p * 4 + 3];

            float[] xyxy = MathOps.cxcywh2xyxy(new float[]{cx, cy, bw, bh});

            xyxy[0] = Math.max(0, Math.min(xyxy[0] * origWidth, origWidth));
            xyxy[1] = Math.max(0, Math.min(xyxy[1] * origHeight, origHeight));
            xyxy[2] = Math.max(0, Math.min(xyxy[2] * origWidth, origWidth));
            xyxy[3] = Math.max(0, Math.min(xyxy[3] * origHeight, origHeight));

            detections.add(new Detection(
                    new BoundingBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    labels.get(bestClass),
                    bestClass,
                    bestScore
            ));
        }

        detections.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));

        return detections;
    }

    // --- Utilities ---

    private static BufferedImage loadImage(Path path) {
        try {
            BufferedImage image = ImageIO.read(path.toFile());
            if (image == null) {
                throw new InferenceException("Unsupported image format: " + path);
            }
            return image;
        } catch (IOException e) {
            throw new InferenceException("Failed to read image: " + path, e);
        }
    }

    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private Labels labels = Labels.coco();
        private String inputName;
        private int inputSize = 640;
        private float confidenceThreshold = 0.5f;

        public Builder session(InferenceSession session) {
            this.session = session;
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

        public Builder labels(Labels labels) {
            this.labels = labels;
            return this;
        }

        public Builder inputName(String inputName) {
            this.inputName = inputName;
            return this;
        }

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
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id);
                loadFromDirectory(dir);
            }
            if (inputName == null) {
                inputName = session.inputNames().iterator().next();
            }
            return new Yolo26Detector(session, labels, inputName, inputSize, confidenceThreshold);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }

            this.session = InferenceSession.create(modelPath);
            try {
                if (this.inputName == null) {
                    this.inputName = session.inputNames().iterator().next();
                }

                Path labelsPath = dir.resolve("labels.txt");
                if (Files.exists(labelsPath)) {
                    this.labels = Labels.fromFile(labelsPath);
                }

                long[] inputShape = session.inputShape(this.inputName);
                ImageLayout layout = ImageLayout.detect(inputShape);
                this.inputSize = layout.imageSize(inputShape);
            } catch (Exception e) {
                this.session.close();
                this.session = null;
                throw e;
            }
        }
    }
}
