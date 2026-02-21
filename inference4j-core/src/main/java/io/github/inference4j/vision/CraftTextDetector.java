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
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * CRAFT (Character Region Awareness for Text Detection) text detector.
 *
 * <p>CRAFT detects text regions in images by producing pixel-level heatmaps:
 * a <em>region score</em> (character centers) and an <em>affinity score</em>
 * (character spacing). These are combined and thresholded to find connected
 * components, which become text region bounding boxes.
 *
 * <p>Unlike YOLO-based detectors, CRAFT does not require Non-Maximum Suppression
 * because connected components are naturally non-overlapping.
 *
 * <h2>Preprocessing</h2>
 * <ul>
 *   <li>Resize maintaining aspect ratio, long side = {@code targetSize}</li>
 *   <li>Both dimensions rounded to nearest multiple of 32 (VGG16 backbone requirement)</li>
 *   <li>ImageNet normalization: {@code (pixel/255 - mean) / std}</li>
 *   <li>Input layout: NCHW {@code [1, 3, H, W]}</li>
 * </ul>
 *
 * <h2>Output handling</h2>
 * <p>CRAFT outputs heatmaps at half the input resolution. The wrapper handles
 * both single combined tensor {@code [1, H/2, W/2, 2]} and two separate tensors.
 *
 * <h2>Threshold tuning</h2>
 * <p>The default thresholds ({@code textThreshold=0.7}, {@code lowTextThreshold=0.4})
 * match the original CRAFT paper and work well for <strong>clean document scans</strong>
 * with high-contrast text. For real-world images (product labels, street signs,
 * photos with embedded text), significantly lower values are needed — try
 * {@code textThreshold=0.4} and {@code lowTextThreshold=0.3} as a starting point.
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (CraftTextDetector detector = CraftTextDetector.builder().build()) {
 *     List<TextRegion> regions = detector.detect(Path.of("document.jpg"));
 *     for (TextRegion r : regions) {
 *         System.out.printf("Text at [%.0f, %.0f, %.0f, %.0f] (confidence=%.2f)%n",
 *             r.box().x1(), r.box().y1(), r.box().x2(), r.box().y2(),
 *             r.confidence());
 *     }
 * }
 * }</pre>
 *
 * @see TextRegion
 * @see TextDetector
 */
public class CraftTextDetector
        extends AbstractInferenceTask<BufferedImage, List<TextRegion>>
        implements TextDetector {

    private static final String DEFAULT_MODEL_ID = "inference4j/craft-mlt-25k";
    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    private final String inputName;
    private final int targetSize;
    private final float defaultTextThreshold;
    private final float defaultLowTextThreshold;
    private final int minComponentArea;

    private CraftTextDetector(InferenceSession session, String inputName, int targetSize,
                              float defaultTextThreshold, float defaultLowTextThreshold,
                              int minComponentArea) {
        super(session,
                createPreprocessor(inputName, targetSize),
                ctx -> decodeTextRegions(ctx.outputs(),
                        defaultTextThreshold, defaultLowTextThreshold, minComponentArea,
                        targetSize, ctx.input().getWidth(), ctx.input().getHeight()));
        this.inputName = inputName;
        this.targetSize = targetSize;
        this.defaultTextThreshold = defaultTextThreshold;
        this.defaultLowTextThreshold = defaultLowTextThreshold;
        this.minComponentArea = minComponentArea;
    }

    /**
     * Creates a builder for custom CRAFT configuration.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    @Override
    public List<TextRegion> detect(BufferedImage image) {
        return run(image);
    }

    @Override
    public List<TextRegion> detect(BufferedImage image, float textThreshold, float lowTextThreshold) {
        Map<String, Tensor> inputs = preprocessor.process(image);
        Map<String, Tensor> outputs = session.run(inputs);
        return decodeTextRegions(outputs, textThreshold, lowTextThreshold,
                minComponentArea, targetSize, image.getWidth(), image.getHeight());
    }

    @Override
    public List<TextRegion> detect(Path imagePath) {
        return detect(imagePath, defaultTextThreshold, defaultLowTextThreshold);
    }

    @Override
    public List<TextRegion> detect(Path imagePath, float textThreshold, float lowTextThreshold) {
        return detect(loadImage(imagePath), textThreshold, lowTextThreshold);
    }

    // --- Preprocessing ---

    private static io.github.inference4j.processing.Preprocessor<BufferedImage, Map<String, Tensor>> createPreprocessor(
            String inputName, int targetSize) {
        return image -> {
            ResizeResult resize = resizeForCraft(image, targetSize);
            Tensor tensor = imageToTensor(resize.image);
            return Map.of(inputName, tensor);
        };
    }

    record ResizeResult(BufferedImage image, float scale) {
    }

    static ResizeResult resizeForCraft(BufferedImage image, int targetSize) {
        int origW = image.getWidth();
        int origH = image.getHeight();

        float scale = (float) targetSize / Math.max(origW, origH);
        int scaledW = roundToMultipleOf32(Math.round(origW * scale));
        int scaledH = roundToMultipleOf32(Math.round(origH * scale));

        BufferedImage result = new BufferedImage(scaledW, scaledH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = result.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(image, 0, 0, scaledW, scaledH, null);
        g.dispose();

        float actualScaleX = (float) scaledW / origW;
        float actualScaleY = (float) scaledH / origH;
        float actualScale = Math.min(actualScaleX, actualScaleY);

        return new ResizeResult(result, actualScale);
    }

    static int roundToMultipleOf32(int value) {
        return Math.max(32, ((value + 31) / 32) * 32);
    }

    static Tensor imageToTensor(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        float[] data = new float[3 * h * w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = image.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255f;
                float g = ((rgb >> 8) & 0xFF) / 255f;
                float b = (rgb & 0xFF) / 255f;

                data[0 * h * w + y * w + x] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                data[1 * h * w + y * w + x] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                data[2 * h * w + y * w + x] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
            }
        }

        return Tensor.fromFloats(data, new long[]{1, 3, h, w});
    }

    // --- Post-processing ---

    private static List<TextRegion> decodeTextRegions(Map<String, Tensor> outputs,
                                                       float textThreshold, float lowTextThreshold,
                                                       int minArea, int targetSize,
                                                       int origW, int origH) {
        Tensor scoreMap = findScoreMap(outputs);
        long[] shape = scoreMap.shape();
        int heatmapH = (int) shape[1];
        int heatmapW = (int) shape[2];
        float[] data = scoreMap.toFloats();

        float[] regionScore = new float[heatmapH * heatmapW];
        float[] affinityScore = new float[heatmapH * heatmapW];
        for (int y = 0; y < heatmapH; y++) {
            for (int x = 0; x < heatmapW; x++) {
                int idx = y * heatmapW * 2 + x * 2;
                regionScore[y * heatmapW + x] = data[idx];
                affinityScore[y * heatmapW + x] = data[idx + 1];
            }
        }

        // Recompute scale from original dimensions
        float rawScale = (float) targetSize / Math.max(origW, origH);
        int scaledW = roundToMultipleOf32(Math.round(origW * rawScale));
        int scaledH = roundToMultipleOf32(Math.round(origH * rawScale));
        float actualScaleX = (float) scaledW / origW;
        float actualScaleY = (float) scaledH / origH;
        float scale = Math.min(actualScaleX, actualScaleY);

        return postProcess(regionScore, affinityScore, heatmapH, heatmapW,
                textThreshold, lowTextThreshold, minArea,
                scale, origW, origH);
    }

    static List<TextRegion> postProcess(float[] regionScore, float[] affinityScore,
                                        int heatmapH, int heatmapW,
                                        float textThreshold, float lowTextThreshold,
                                        int minArea, float scale,
                                        int origW, int origH) {
        float[] combined = new float[heatmapH * heatmapW];
        for (int i = 0; i < combined.length; i++) {
            combined[i] = Math.min(1f, Math.max(0f, regionScore[i] + affinityScore[i]));
        }

        boolean[] binary = new boolean[combined.length];
        for (int i = 0; i < combined.length; i++) {
            binary[i] = combined[i] >= lowTextThreshold;
        }

        int[] labels = connectedComponents(binary, heatmapW, heatmapH);

        int maxLabel = 0;
        for (int label : labels) {
            if (label > maxLabel) {
                maxLabel = label;
            }
        }

        if (maxLabel == 0) {
            return List.of();
        }

        int[] minX = new int[maxLabel + 1];
        int[] minY = new int[maxLabel + 1];
        int[] maxX = new int[maxLabel + 1];
        int[] maxY = new int[maxLabel + 1];
        float[] scoreSum = new float[maxLabel + 1];
        int[] area = new int[maxLabel + 1];

        for (int i = 1; i <= maxLabel; i++) {
            minX[i] = Integer.MAX_VALUE;
            minY[i] = Integer.MAX_VALUE;
            maxX[i] = Integer.MIN_VALUE;
            maxY[i] = Integer.MIN_VALUE;
        }

        for (int y = 0; y < heatmapH; y++) {
            for (int x = 0; x < heatmapW; x++) {
                int label = labels[y * heatmapW + x];
                if (label == 0) continue;

                area[label]++;
                scoreSum[label] += regionScore[y * heatmapW + x];
                if (x < minX[label]) minX[label] = x;
                if (y < minY[label]) minY[label] = y;
                if (x > maxX[label]) maxX[label] = x;
                if (y > maxY[label]) maxY[label] = y;
            }
        }

        List<TextRegion> regions = new ArrayList<>();
        for (int i = 1; i <= maxLabel; i++) {
            if (area[i] < minArea) continue;

            float meanScore = scoreSum[i] / area[i];
            if (meanScore < textThreshold) continue;

            float x1 = Math.max(0, Math.min(minX[i] * 2f / scale, origW));
            float y1 = Math.max(0, Math.min(minY[i] * 2f / scale, origH));
            float x2 = Math.max(0, Math.min((maxX[i] + 1) * 2f / scale, origW));
            float y2 = Math.max(0, Math.min((maxY[i] + 1) * 2f / scale, origH));

            regions.add(new TextRegion(new BoundingBox(x1, y1, x2, y2), meanScore));
        }

        regions.sort(Comparator.comparingDouble(TextRegion::confidence).reversed());
        return regions;
    }

    static int[] connectedComponents(boolean[] binary, int width, int height) {
        int[] labels = new int[width * height];
        int currentLabel = 0;

        int[] queueX = new int[width * height];
        int[] queueY = new int[width * height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                if (!binary[idx] || labels[idx] != 0) continue;

                currentLabel++;
                int head = 0;
                int tail = 0;
                queueX[tail] = x;
                queueY[tail] = y;
                tail++;
                labels[idx] = currentLabel;

                while (head < tail) {
                    int cx = queueX[head];
                    int cy = queueY[head];
                    head++;

                    int[] dx = {0, 0, -1, 1};
                    int[] dy = {-1, 1, 0, 0};

                    for (int d = 0; d < 4; d++) {
                        int nx = cx + dx[d];
                        int ny = cy + dy[d];
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        int nIdx = ny * width + nx;
                        if (!binary[nIdx] || labels[nIdx] != 0) continue;

                        labels[nIdx] = currentLabel;
                        queueX[tail] = nx;
                        queueY[tail] = ny;
                        tail++;
                    }
                }
            }
        }

        return labels;
    }

    // --- Utilities ---

    private static Tensor findScoreMap(Map<String, Tensor> outputs) {
        Tensor scoreMap = outputs.get("score_map");
        if (scoreMap != null) {
            return scoreMap;
        }

        for (Tensor tensor : outputs.values()) {
            long[] shape = tensor.shape();
            if (shape.length == 4 && shape[3] == 2) {
                return tensor;
            }
        }

        return outputs.values().iterator().next();
    }

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

    /**
     * Builder for custom CRAFT configuration.
     */
    public static class Builder {
        private InferenceSession session;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;
        private String inputName;
        private int targetSize = 1280;
        private float textThreshold = 0.7f;
        private float lowTextThreshold = 0.4f;
        private int minComponentArea = 10;

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

        public Builder inputName(String inputName) {
            this.inputName = inputName;
            return this;
        }

        public Builder targetSize(int targetSize) {
            this.targetSize = targetSize;
            return this;
        }

        public Builder textThreshold(float textThreshold) {
            this.textThreshold = textThreshold;
            return this;
        }

        public Builder lowTextThreshold(float lowTextThreshold) {
            this.lowTextThreshold = lowTextThreshold;
            return this;
        }

        public Builder minComponentArea(int minComponentArea) {
            this.minComponentArea = minComponentArea;
            return this;
        }

        public CraftTextDetector build() {
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
            return new CraftTextDetector(session, inputName, targetSize,
                    textThreshold, lowTextThreshold, minComponentArea);
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
        }
    }
}
