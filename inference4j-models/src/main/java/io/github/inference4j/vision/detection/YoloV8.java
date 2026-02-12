package io.github.inference4j.vision.detection;

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
 * YOLOv8 object detection model wrapper.
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
 * try (YoloV8 yolo = YoloV8.fromPretrained("models/yolov8n")) {
 *     List<Detection> detections = yolo.detect(Path.of("street.jpg"));
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
 * try (YoloV8 yolo = YoloV8.builder()
 *         .session(InferenceSession.create(modelPath))
 *         .labels(Labels.fromFile(Path.of("custom-labels.txt")))
 *         .inputSize(640)
 *         .confidenceThreshold(0.5f)
 *         .iouThreshold(0.4f)
 *         .build()) {
 *     List<Detection> detections = yolo.detect(image);
 * }
 * }</pre>
 *
 * @see Detection
 * @see BoundingBox
 * @see Yolo26
 */
public class YoloV8 implements ObjectDetectionModel {

    private static final float LETTERBOX_PAD_VALUE = 114f / 255f;

    private final InferenceSession session;
    private final Labels labels;
    private final String inputName;
    private final int inputSize;
    private final float defaultConfidenceThreshold;
    private final float defaultIouThreshold;

    private YoloV8(InferenceSession session, Labels labels, String inputName,
                 int inputSize, float defaultConfidenceThreshold, float defaultIouThreshold) {
        this.session = session;
        this.labels = labels;
        this.inputName = inputName;
        this.inputSize = inputSize;
        this.defaultConfidenceThreshold = defaultConfidenceThreshold;
        this.defaultIouThreshold = defaultIouThreshold;
    }

    public static YoloV8 fromPretrained(String modelPath) {
        Path dir = Path.of(modelPath);
        return fromModelDirectory(dir);
    }

    public static YoloV8 fromPretrained(String modelId, ModelSource source) {
        Path dir = source.resolve(modelId);
        return fromModelDirectory(dir);
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<Detection> detect(BufferedImage image) {
        return detect(image, defaultConfidenceThreshold, defaultIouThreshold);
    }

    @Override
    public List<Detection> detect(BufferedImage image, float confidenceThreshold, float iouThreshold) {
        int origW = image.getWidth();
        int origH = image.getHeight();

        LetterboxResult lb = letterbox(image, inputSize);
        Tensor inputTensor = imageToTensor(lb.image);

        Map<String, Tensor> inputs = new LinkedHashMap<>();
        inputs.put(inputName, inputTensor);

        Map<String, Tensor> outputs = session.run(inputs);
        Tensor outputTensor = outputs.values().iterator().next();

        return postProcess(outputTensor.toFloats(), outputTensor.shape(),
                labels, confidenceThreshold, iouThreshold,
                lb.scale, lb.padX, lb.padY, origW, origH);
    }

    @Override
    public List<Detection> detect(Path imagePath) {
        return detect(imagePath, defaultConfidenceThreshold, defaultIouThreshold);
    }

    @Override
    public List<Detection> detect(Path imagePath, float confidenceThreshold, float iouThreshold) {
        return detect(loadImage(imagePath), confidenceThreshold, iouThreshold);
    }

    @Override
    public void close() {
        session.close();
    }

    // --- Preprocessing ---

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

    /**
     * Post-processes raw YOLOv8 output into a list of detections.
     *
     * <p>Package-visible for unit testing without an ONNX session.
     *
     * @param rawOutput   flat float array from model output
     * @param outputShape shape of the output tensor (e.g., {@code [1, 84, 8400]})
     * @param labels      class labels
     * @param confThreshold minimum confidence to keep a detection
     * @param iouThreshold  IoU threshold for NMS
     * @param scale       letterbox scale factor
     * @param padX        horizontal padding added by letterbox
     * @param padY        vertical padding added by letterbox
     * @param origWidth   original image width
     * @param origHeight  original image height
     * @return detections sorted by confidence descending
     */
    static List<Detection> postProcess(float[] rawOutput, long[] outputShape,
                                       Labels labels,
                                       float confThreshold, float iouThreshold,
                                       float scale, float padX, float padY,
                                       int origWidth, int origHeight) {
        // YOLOv8 output: [1, numOutputs, numCandidates] where numOutputs = 4 + numClasses
        int numOutputs = (int) outputShape[1];
        int numCandidates = (int) outputShape[2];
        int numClasses = numOutputs - 4;

        // Collect candidates above confidence threshold
        List<float[]> boxList = new ArrayList<>();
        List<Float> scoreList = new ArrayList<>();
        List<Integer> classList = new ArrayList<>();

        for (int c = 0; c < numCandidates; c++) {
            // Find best class for this candidate
            int bestClass = -1;
            float bestScore = -1f;
            for (int cls = 0; cls < numClasses; cls++) {
                // Output is [1, numOutputs, numCandidates] — row-major:
                // element at [0, row, col] = rawOutput[row * numCandidates + col]
                float score = rawOutput[(4 + cls) * numCandidates + c];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = cls;
                }
            }

            if (bestScore < confThreshold) {
                continue;
            }

            // Extract box [cx, cy, w, h]
            float cx = rawOutput[0 * numCandidates + c];
            float cy = rawOutput[1 * numCandidates + c];
            float bw = rawOutput[2 * numCandidates + c];
            float bh = rawOutput[3 * numCandidates + c];

            // Convert to [x1, y1, x2, y2]
            float[] xyxy = MathOps.cxcywh2xyxy(new float[]{cx, cy, bw, bh});

            // Rescale to original image coordinates
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

        // Flatten boxes for NMS
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

    private static YoloV8 fromModelDirectory(Path dir) {
        if (!Files.isDirectory(dir)) {
            throw new ModelSourceException("Model directory not found: " + dir);
        }

        Path modelPath = dir.resolve("model.onnx");
        if (!Files.exists(modelPath)) {
            throw new ModelSourceException("Model file not found: " + modelPath);
        }

        InferenceSession session = InferenceSession.create(modelPath);
        try {
            String inputName = session.inputNames().iterator().next();

            Path labelsPath = dir.resolve("labels.txt");
            Labels labels = Files.exists(labelsPath) ? Labels.fromFile(labelsPath) : Labels.coco();

            long[] inputShape = session.inputShape(inputName);
            ImageLayout layout = ImageLayout.detect(inputShape);
            int inputSize = layout.imageSize(inputShape);

            return new YoloV8(session, labels, inputName, inputSize, 0.25f, 0.45f);
        } catch (Exception e) {
            session.close();
            throw e;
        }
    }

    public static class Builder {
        private InferenceSession session;
        private Labels labels = Labels.coco();
        private String inputName;
        private int inputSize = 640;
        private float confidenceThreshold = 0.25f;
        private float iouThreshold = 0.45f;

        public Builder session(InferenceSession session) {
            this.session = session;
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

        public YoloV8 build() {
            if (session == null) {
                throw new IllegalStateException("InferenceSession is required");
            }
            if (inputName == null) {
                inputName = session.inputNames().iterator().next();
            }
            return new YoloV8(session, labels, inputName, inputSize,
                    confidenceThreshold, iouThreshold);
        }
    }
}
