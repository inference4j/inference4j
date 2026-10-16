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
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Text line detector backed by PaddleOCR's DB (Differentiable Binarization) detection models.
 *
 * <p>Finds lines of text, including rotated ones, and returns them in reading order: top to bottom, then left
 * to right. Each {@link TextRegion} carries the line's four-corner outline, which {@link QuadCrop} turns into an
 * upright image for a {@link TextRecognizer}.
 *
 * <h2>Presets</h2>
 * <pre>{@code
 * try (var detector = PaddleTextDetector.small().build()) {        // PP-OCRv6 small, 9.9 MB (default)
 *     List<TextRegion> lines = detector.detect(Path.of("receipt.jpg"));
 * }
 * try (var detector = PaddleTextDetector.tiny().build()) { ... }   // PP-OCRv6 tiny, 1.8 MB
 * }</pre>
 *
 * <p>Defaults follow the {@code paddleocr} package's OCR pipeline: images keep their own size (rounded to
 * multiples of 32, at most 4000 pixels on the long side, at least 64 on the short side), and DB post-processing
 * uses {@code threshold 0.3}, {@code boxThreshold 0.6} and {@code unclipRatio 1.5}.
 *
 * @see OcrPipeline
 * @see DbPostProcessor
 */
public class PaddleTextDetector implements TextDetector {

    private static final String SMALL_MODEL_ID = "inference4j/pp-ocrv6-small-det";
    private static final String TINY_MODEL_ID = "inference4j/pp-ocrv6-tiny-det";
    // ImageNet statistics. The models take RGB: paddleocr's predictors convert to RGB although inference.yml says BGR
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private final InferenceSession session;
    private final String inputName;
    private final DbPostProcessor postProcessor;
    private final int minSide;
    private final int maxSide;

    private PaddleTextDetector(InferenceSession session, DbPostProcessor postProcessor, int minSide, int maxSide) {
        this.session = session;
        this.inputName = session.inputNames().iterator().next();
        this.postProcessor = postProcessor;
        this.minSide = minSide;
        this.maxSide = maxSide;
    }

    /** PP-OCRv6 small detection model (9.9 MB), the default. */
    public static Builder small() {
        return new Builder().modelId(SMALL_MODEL_ID);
    }

    /** PP-OCRv6 tiny detection model (1.8 MB). */
    public static Builder tiny() {
        return new Builder().modelId(TINY_MODEL_ID);
    }

    /** A builder for the default model; set {@link Builder#modelId(String)} for another PaddleOCR DB detector. */
    public static Builder builder() {
        return small();
    }

    @Override
    public List<TextRegion> detect(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int[] pixels = PaddleImages.pixels(image);
        int paddedWidth = width;
        int paddedHeight = height;
        if (width + height < 64) {
            // PaddleOCR pads tiny images to at least 32 × 32 with black
            paddedWidth = Math.max(32, width);
            paddedHeight = Math.max(32, height);
            int[] padded = new int[paddedWidth * paddedHeight];
            for (int y = 0; y < height; y++) {
                System.arraycopy(pixels, y * width, padded, y * paddedWidth, width);
            }
            pixels = padded;
        }

        int[] size = resizedSize(paddedWidth, paddedHeight);
        int resizedWidth = size[0];
        int resizedHeight = size[1];
        int[] resized = PaddleImages.resize(pixels, paddedWidth, paddedHeight, resizedWidth, resizedHeight);

        int plane = resizedWidth * resizedHeight;
        float[] input = new float[3 * plane];
        for (int i = 0; i < plane; i++) {
            int rgb = resized[i];
            float r = ((rgb >> 16) & 0xFF) / 255f;
            float g = ((rgb >> 8) & 0xFF) / 255f;
            float b = (rgb & 0xFF) / 255f;
            input[i] = (r - MEAN[0]) / STD[0];
            input[plane + i] = (g - MEAN[1]) / STD[1];
            input[2 * plane + i] = (b - MEAN[2]) / STD[2];
        }
        Tensor tensor = Tensor.fromFloats(input, new long[]{1, 3, resizedHeight, resizedWidth});
        Tensor output = session.run(Map.of(inputName, tensor)).values().iterator().next();
        long[] shape = output.shape();
        int mapHeight = (int) shape[shape.length - 2];
        int mapWidth = (int) shape[shape.length - 1];

        List<TextRegion> regions = postProcessor.process(output.toFloats(), mapWidth, mapHeight, width, height);
        return ReadingOrder.sort(regions);
    }

    @Override
    public List<TextRegion> detect(Path imagePath) {
        return detect(PaddleImages.read(imagePath));
    }

    @Override
    public void close() {
        session.close();
    }

    /** PaddleOCR's {@code DetResizeForTest} with {@code limit_type: min}: multiples of 32, within the limits. */
    private int[] resizedSize(int width, int height) {
        double ratio = Math.min(width, height) < minSide ? (double) minSide / Math.min(width, height) : 1.0;
        int resizedHeight = (int) (height * ratio);
        int resizedWidth = (int) (width * ratio);
        if (Math.max(resizedHeight, resizedWidth) > maxSide) {
            double shrink = (double) maxSide / Math.max(resizedHeight, resizedWidth);
            resizedHeight = (int) (resizedHeight * shrink);
            resizedWidth = (int) (resizedWidth * shrink);
        }
        // Math.rint rounds halves to even, like Python's round()
        resizedHeight = Math.max((int) (Math.rint(resizedHeight / 32.0) * 32), 32);
        resizedWidth = Math.max((int) (Math.rint(resizedWidth / 32.0) * 32), 32);
        return new int[]{resizedWidth, resizedHeight};
    }

    public static class Builder {
        private ModelSource modelSource;
        private String modelId = SMALL_MODEL_ID;
        private SessionConfigurer sessionConfigurer;
        private float threshold = 0.3f;
        private float boxThreshold = 0.6f;
        private float unclipRatio = 1.5f;
        private int maxCandidates = 3000;
        private int minSide = 64;
        private int maxSide = 4000;

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

        /** Probability above which a pixel of the model's map counts as text. Defaults to 0.3. */
        public Builder threshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        /** Minimum mean probability inside a line for it to be kept. Defaults to 0.6. */
        public Builder boxThreshold(float boxThreshold) {
            this.boxThreshold = boxThreshold;
            return this;
        }

        /** How far detected lines are expanded beyond the model's shrunken text core. Defaults to 1.5. */
        public Builder unclipRatio(float unclipRatio) {
            this.unclipRatio = unclipRatio;
            return this;
        }

        /** Maximum number of candidate areas examined per image. Defaults to 3000. */
        public Builder maxCandidates(int maxCandidates) {
            this.maxCandidates = maxCandidates;
            return this;
        }

        /** Images with a shorter side below this are scaled up to it. Defaults to 64 pixels. */
        public Builder minSide(int minSide) {
            this.minSide = minSide;
            return this;
        }

        /** Images with a longer side above this are scaled down to it. Defaults to 4000 pixels. */
        public Builder maxSide(int maxSide) {
            this.maxSide = maxSide;
            return this;
        }

        public PaddleTextDetector build() {
            if (minSide < 1 || maxSide < 32) {
                throw new IllegalArgumentException("minSide must be positive and maxSide at least 32");
            }
            ModelSource source = modelSource != null ? modelSource : HuggingFaceModelSource.defaultInstance();
            Path dir = source.resolve(modelId, List.of("model.onnx"));
            Path modelPath = dir.resolve("model.onnx");
            if (!Files.exists(modelPath)) {
                throw new ModelSourceException("Model file not found: " + modelPath);
            }
            InferenceSession session = sessionConfigurer != null
                    ? InferenceSession.create(modelPath, sessionConfigurer)
                    : InferenceSession.create(modelPath);
            return new PaddleTextDetector(session,
                    new DbPostProcessor(threshold, boxThreshold, unclipRatio, maxCandidates), minSide, maxSide);
        }
    }
}
