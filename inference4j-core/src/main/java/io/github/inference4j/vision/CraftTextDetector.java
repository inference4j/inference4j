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
import io.github.inference4j.session.SessionConfigurer;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * CRAFT (Character Region Awareness for Text Detection) text detector.
 *
 * <p>CRAFT produces two score maps at half the input resolution: a <em>text</em> (region) score for character
 * centers and a <em>link</em> (affinity) score for the spacing between characters of a word. Post-processing
 * follows the reference implementation ({@code clovaai/CRAFT-pytorch}); see {@link CraftPostProcessor}. Regions
 * are words or groups of words, with a four-corner outline that is rotated for rotated text.
 *
 * <h2>Preprocessing</h2>
 * <ul>
 *   <li>Scaled up by at most {@code magRatio} (1.5), with the long side at most {@code targetSize} (1280),
 *       keeping the aspect ratio</li>
 *   <li>Padded with black to multiples of 32 (VGG16 backbone requirement)</li>
 *   <li>ImageNet normalization: {@code (pixel/255 - mean) / std}, layout NCHW {@code [1, 3, H, W]}</li>
 * </ul>
 *
 * <h2>Thresholds</h2>
 * <p>The defaults are the reference's: {@code textThreshold 0.7}, {@code linkThreshold 0.4},
 * {@code lowTextThreshold 0.4}. Lower {@code textThreshold} to find fainter text, raise {@code linkThreshold} to
 * split words that merge.
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
 * @see CraftPostProcessor
 */
public class CraftTextDetector
        extends AbstractInferenceTask<BufferedImage, List<TextRegion>>
        implements TextDetector {

    private static final String DEFAULT_MODEL_ID = "inference4j/craft-mlt-25k";
    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    private final int targetSize;
    private final float magRatio;
    private final float defaultTextThreshold;
    private final float defaultLowTextThreshold;
    private final float linkThreshold;
    private final int minComponentArea;

    private CraftTextDetector(InferenceSession session, String inputName, Builder builder) {
        super(session,
                image -> PreprocessResult.of(Map.of(inputName, toTensor(resize(image, builder.targetSize, builder.magRatio)))),
                ctx -> decode(ctx.outputs(), new CraftPostProcessor(builder.textThreshold, builder.linkThreshold,
                                builder.lowTextThreshold, builder.minComponentArea),
                        ctx.input(), builder.targetSize, builder.magRatio));
        this.targetSize = builder.targetSize;
        this.magRatio = builder.magRatio;
        this.defaultTextThreshold = builder.textThreshold;
        this.defaultLowTextThreshold = builder.lowTextThreshold;
        this.linkThreshold = builder.linkThreshold;
        this.minComponentArea = builder.minComponentArea;
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

    /** Detects text with text thresholds other than the configured ones; the link threshold stays as configured. */
    public List<TextRegion> detect(BufferedImage image, float textThreshold, float lowTextThreshold) {
        PreprocessResult result = preprocessor.process(image);
        Map<String, Tensor> outputs = session.run(result.tensors());
        return decode(outputs, new CraftPostProcessor(textThreshold, linkThreshold, lowTextThreshold, minComponentArea),
                image, targetSize, magRatio);
    }

    @Override
    public List<TextRegion> detect(Path imagePath) {
        return detect(imagePath, defaultTextThreshold, defaultLowTextThreshold);
    }

    /** Detects text with text thresholds other than the configured ones; the link threshold stays as configured. */
    public List<TextRegion> detect(Path imagePath, float textThreshold, float lowTextThreshold) {
        return detect(PaddleImages.read(imagePath), textThreshold, lowTextThreshold);
    }

    /** The resized image, padded to multiples of 32: as the reference's {@code resize_aspect_ratio}. */
    private record Resized(int[] pixels, int width, int height, double ratio) {
    }

    private static Resized resize(BufferedImage image, int targetSize, float magRatio) {
        int width = image.getWidth();
        int height = image.getHeight();
        double ratio = scaleRatio(width, height, targetSize, magRatio);
        int targetW = (int) (width * ratio);
        int targetH = (int) (height * ratio);
        int[] resized = PaddleImages.resize(PaddleImages.pixels(image), width, height, targetW, targetH);
        int paddedW = targetW % 32 == 0 ? targetW : targetW + (32 - targetW % 32);
        int paddedH = targetH % 32 == 0 ? targetH : targetH + (32 - targetH % 32);
        int[] padded = new int[paddedW * paddedH];   // black
        for (int y = 0; y < targetH; y++) {
            System.arraycopy(resized, y * targetW, padded, y * paddedW, targetW);
        }
        return new Resized(padded, paddedW, paddedH, ratio);
    }

    private static double scaleRatio(int width, int height, int targetSize, float magRatio) {
        double target = (double) magRatio * Math.max(width, height);
        if (target > targetSize) {
            target = targetSize;
        }
        return target / Math.max(width, height);
    }

    private static Tensor toTensor(Resized image) {
        int plane = image.width() * image.height();
        float[] data = new float[3 * plane];
        float[] mean = new float[3];
        float[] std = new float[3];
        for (int c = 0; c < 3; c++) {
            mean[c] = (float) (IMAGENET_MEAN[c] * 255.0);
            std[c] = (float) (IMAGENET_STD[c] * 255.0);
        }
        for (int i = 0; i < plane; i++) {
            int rgb = image.pixels()[i];
            data[i] = (((rgb >> 16) & 0xFF) - mean[0]) / std[0];
            data[plane + i] = (((rgb >> 8) & 0xFF) - mean[1]) / std[1];
            data[2 * plane + i] = ((rgb & 0xFF) - mean[2]) / std[2];
        }
        return Tensor.fromFloats(data, new long[]{1, 3, image.height(), image.width()});
    }

    private static List<TextRegion> decode(Map<String, Tensor> outputs, CraftPostProcessor postProcessor,
                                           BufferedImage image, int targetSize, float magRatio) {
        Tensor scoreMap = findScoreMap(outputs);
        long[] shape = scoreMap.shape();
        int mapHeight = (int) shape[1];
        int mapWidth = (int) shape[2];
        float[] data = scoreMap.toFloats();
        float[] text = new float[mapHeight * mapWidth];
        float[] link = new float[mapHeight * mapWidth];
        for (int i = 0; i < text.length; i++) {
            text[i] = data[2 * i];
            link[i] = data[2 * i + 1];
        }
        double ratio = scaleRatio(image.getWidth(), image.getHeight(), targetSize, magRatio);
        // The reference multiplies by (1 / ratio) * 2: each score-map pixel covers two input pixels
        double scale = (1 / ratio) * 2;
        return postProcessor.process(text, link, mapWidth, mapHeight, scale, image.getWidth(), image.getHeight());
    }

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
        private float magRatio = 1.5f;
        private float textThreshold = 0.7f;
        private float linkThreshold = 0.4f;
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

        /** Maximum long side of the model input, in pixels. Defaults to 1280. */
        public Builder targetSize(int targetSize) {
            this.targetSize = targetSize;
            return this;
        }

        /** How much smaller images are scaled up, at most (still within {@code targetSize}). Defaults to 1.5. */
        public Builder magRatio(float magRatio) {
            this.magRatio = magRatio;
            return this;
        }

        /** Minimum highest text score for an area to count as text. Defaults to 0.7. */
        public Builder textThreshold(float textThreshold) {
            this.textThreshold = textThreshold;
            return this;
        }

        /** Link score above which characters are joined into words. Defaults to 0.4. */
        public Builder linkThreshold(float linkThreshold) {
            this.linkThreshold = linkThreshold;
            return this;
        }

        /** Text score above which a pixel belongs to a character. Defaults to 0.4. */
        public Builder lowTextThreshold(float lowTextThreshold) {
            this.lowTextThreshold = lowTextThreshold;
            return this;
        }

        /** Minimum area in score-map pixels (half the input resolution). Defaults to 10. */
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
            return new CraftTextDetector(session, inputName, this);
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
