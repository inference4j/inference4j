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
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.processing.MathOps;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.preprocessing.image.ImageLayout;
import io.github.inference4j.preprocessing.image.Labels;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
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
 *         .modelSource(ModelSource.fromPath(localDir))
 *         .sessionOptions(opts -> opts.addCUDA(0))
 *         .labels(Labels.fromFile(Path.of("custom-labels.txt")))
 *         .confidenceThreshold(0.5f)
 *         .iouThreshold(0.4f)
 *         .build()) {
 *     List<Detection> detections = detector.detect(image);
 * }
 * }</pre>
 *
 * @see Detection
 * @see BoundingBox
 * @see Yolo26Detector
 */
public class YoloV8Detector
        extends AbstractInferenceTask<BufferedImage, List<Detection>>
        implements ObjectDetector {

    private static final String DEFAULT_MODEL_ID = "inference4j/yolov8n";
    private static final float LETTERBOX_PAD_VALUE = 114f / 255f;

    private final Labels labels;
    private final String inputName;
    private final int inputSize;
    private final float defaultConfidenceThreshold;
    private final float defaultIouThreshold;

    private YoloV8Detector(InferenceSession session, Labels labels, String inputName,
                           int inputSize, float defaultConfidenceThreshold, float defaultIouThreshold) {
        super(session,
                createPreprocessor(inputName, inputSize),
                ctx -> decodeDetections(ctx.outputs(), labels,
                        defaultConfidenceThreshold, defaultIouThreshold,
                        inputSize, ctx.input().getWidth(), ctx.input().getHeight()));
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
        Map<String, Tensor> inputs = preprocessor.process(image);
        Map<String, Tensor> outputs = session.run(inputs);
        return decodeDetections(outputs, labels, confidenceThreshold, iouThreshold,
                inputSize, image.getWidth(), image.getHeight());
    }

    @Override
    public List<Detection> detect(Path imagePath) {
        return detect(imagePath, defaultConfidenceThreshold, defaultIouThreshold);
    }

    @Override
    public List<Detection> detect(Path imagePath, float confidenceThreshold, float iouThreshold) {
        return detect(loadImage(imagePath), confidenceThreshold, iouThreshold);
    }

    // --- Preprocessing ---

    private static io.github.inference4j.processing.Preprocessor<BufferedImage, Map<String, Tensor>> createPreprocessor(
            String inputName, int inputSize) {
        return image -> {
            LetterboxResult lb = letterbox(image, inputSize);
            Tensor tensor = imageToTensor(lb.image);
            return Map.of(inputName, tensor);
        };
    }

    record LetterboxResult(BufferedImage image, float scale, float padX, float padY) {
    }

    static LetterboxResult letterbox(BufferedImage image, int targetSize) {
        int origW = image.getWidth();
        int origH = image.getHeight();

        float scale = Math.min((float) targetSize / origW, (float) targetSize / origH);
        int scaledW = Math.round(origW * scale);
        int scaledH = Math.round(origH * scale);

        float padX = (targetSize - scaledW) / 2f;
        float padY = (targetSize - scaledH) / 2f;

        BufferedImage result = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = result.createGraphics();

        // Fill with gray padding
        int padGray = Math.round(114f);
        g.setColor(new Color(padGray, padGray, padGray));
        g.fillRect(0, 0, targetSize, targetSize);

        // Draw scaled image centered
        g.drawImage(image, Math.round(padX), Math.round(padY), scaledW, scaledH, null);
        g.dispose();

        return new LetterboxResult(result, scale, padX, padY);
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

    private static List<Detection> decodeDetections(Map<String, Tensor> outputs,
                                                     Labels labels,
                                                     float confThreshold, float iouThreshold,
                                                     int inputSize,
                                                     int origWidth, int origHeight) {
        Tensor outputTensor = outputs.values().iterator().next();

        float scale = Math.min((float) inputSize / origWidth, (float) inputSize / origHeight);
        int scaledW = Math.round(origWidth * scale);
        int scaledH = Math.round(origHeight * scale);
        float padX = (inputSize - scaledW) / 2f;
        float padY = (inputSize - scaledH) / 2f;

        return postProcess(outputTensor.toFloats(), outputTensor.shape(),
                labels, confThreshold, iouThreshold,
                scale, padX, padY, origWidth, origHeight);
    }

    static List<Detection> postProcess(float[] rawOutput, long[] outputShape,
                                       Labels labels,
                                       float confThreshold, float iouThreshold,
                                       float scale, float padX, float padY,
                                       int origWidth, int origHeight) {
        int numOutputs = (int) outputShape[1];
        int numCandidates = (int) outputShape[2];
        int numClasses = numOutputs - 4;

        List<float[]> boxList = new ArrayList<>();
        List<Float> scoreList = new ArrayList<>();
        List<Integer> classList = new ArrayList<>();

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

            float cx = rawOutput[0 * numCandidates + c];
            float cy = rawOutput[1 * numCandidates + c];
            float bw = rawOutput[2 * numCandidates + c];
            float bh = rawOutput[3 * numCandidates + c];

            float[] xyxy = MathOps.cxcywh2xyxy(new float[]{cx, cy, bw, bh});

            xyxy[0] = Math.max(0, Math.min((xyxy[0] - padX) / scale, origWidth));
            xyxy[1] = Math.max(0, Math.min((xyxy[1] - padY) / scale, origHeight));
            xyxy[2] = Math.max(0, Math.min((xyxy[2] - padX) / scale, origWidth));
            xyxy[3] = Math.max(0, Math.min((xyxy[3] - padY) / scale, origHeight));

            boxList.add(xyxy);
            scoreList.add(bestScore);
            classList.add(bestClass);
        }

        if (boxList.isEmpty()) {
            return List.of();
        }

        float[] allBoxes = new float[boxList.size() * 4];
        float[] allScores = new float[scoreList.size()];
        for (int i = 0; i < boxList.size(); i++) {
            System.arraycopy(boxList.get(i), 0, allBoxes, i * 4, 4);
            allScores[i] = scoreList.get(i);
        }

        int[] kept = MathOps.nms(allBoxes, allScores, iouThreshold);

        List<Detection> detections = new ArrayList<>(kept.length);
        for (int idx : kept) {
            float[] box = boxList.get(idx);
            detections.add(new Detection(
                    new BoundingBox(box[0], box[1], box[2], box[3]),
                    labels.get(classList.get(idx)),
                    classList.get(idx),
                    scoreList.get(idx)
            ));
        }

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
        private SessionConfigurer sessionConfigurer;
        private Labels labels = Labels.coco();
        private String inputName;
        private int inputSize = 640;
        private float confidenceThreshold = 0.25f;
        private float iouThreshold = 0.45f;

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

        public Builder iouThreshold(float iouThreshold) {
            this.iouThreshold = iouThreshold;
            return this;
        }

        public YoloV8Detector build() {
            if (session == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                Path dir = source.resolve(id, List.of("model.onnx"));
                loadFromDirectory(dir);
            }
            if (inputName == null) {
                inputName = session.inputNames().iterator().next();
            }
            return new YoloV8Detector(session, labels, inputName, inputSize,
                    confidenceThreshold, iouThreshold);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }

            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }

            this.session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
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
