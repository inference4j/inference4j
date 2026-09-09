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

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import io.github.inference4j.AbstractInferenceTask;
import io.github.inference4j.InferenceSession;
import io.github.inference4j.PreprocessResult;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.processing.Preprocessor;
import io.github.inference4j.session.SessionConfigurer;

/**
 * Monocular depth estimation with Depth Anything V2.
 *
 * <p>Predicts a relative depth value for every pixel of a single ordinary photograph.
 * The result is a {@link DepthMap} the same size as the input image.
 *
 * <h2>Tested model</h2>
 * <p><a href="https://huggingface.co/inference4j/depth-anything-v2-small">inference4j/depth-anything-v2-small</a>
 * — the Small variant (~25M parameters), a good balance of quality and CPU speed.
 * Base and Large variants use the same graph and can be swapped via {@code .modelId(...)}.
 *
 * <p>Other DPT-family exports may work, but output tensor rank varies between exports.
 * This wrapper accepts both {@code [1, H, W]} and {@code [1, 1, H, W]}.
 *
 * <h2>Preprocessing</h2>
 * <ul>
 *   <li>Bicubic resize to {@code targetSize} on the longer edge, aspect ratio preserved</li>
 *   <li>Both dimensions rounded up to a multiple of 14 — the model's patch size</li>
 *   <li>ImageNet mean/std normalization, NCHW layout</li>
 * </ul>
 *
 * <p>Depth is predicted at the model's working resolution and resampled back to the
 * original image dimensions, so {@link DepthMap} coordinates always match the input.
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (var estimator = DepthAnythingEstimator.builder().build()) {
 *     DepthMap depth = estimator.estimate(Path.of("room.jpg"));
 *     ImageAnnotator.save(depth.toImage(Colormap.TURBO), Path.of("depth.png"));
 * }
 * }</pre>
 *
 * <h2>Custom configuration</h2>
 * <pre>{@code
 * try (var estimator = DepthAnythingEstimator.builder()
 *         .modelId("inference4j/depth-anything-v2-base")
 *         .targetSize(700)
 *         .sessionOptions(opts -> opts.addCoreML())
 *         .build()) {
 *     DepthMap depth = estimator.estimate(image);
 * }
 * }</pre>
 *
 * @see DepthEstimator
 * @see DepthMap
 */
public class DepthAnythingEstimator extends AbstractInferenceTask<BufferedImage, DepthMap>
        implements DepthEstimator {

    private static final String DEFAULT_MODEL_ID = "inference4j/depth-anything-v2-small";
    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    /** Depth Anything uses a ViT patch size of 14, so input dimensions must be a multiple of it. */
    private static final int PATCH_MULTIPLE = 14;

    private static final int DEFAULT_TARGET_SIZE = 518;

    private final String inputName;
    private final int targetSize;

    private DepthAnythingEstimator(InferenceSession session, String inputName, int targetSize) {
        super(session,
                createPreprocessor(inputName, targetSize),
                ctx -> decodeDepth(ctx.outputs(),
                        ctx.input().getWidth(), ctx.input().getHeight()));
        this.inputName = inputName;
        this.targetSize = targetSize;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public DepthMap estimate(BufferedImage image) {
        return run(image);
    }

    @Override
    public DepthMap estimate(Path imagePath) {
        return estimate(loadImage(imagePath));
    }

    // --- Preprocessing ---

    private static Preprocessor<BufferedImage, PreprocessResult> createPreprocessor(
            String inputName, int targetSize) {
        return image -> {
            BufferedImage resized = resizeForModel(image, targetSize);
            return PreprocessResult.of(Map.of(inputName, imageToTensor(resized)));
        };
    }

    static BufferedImage resizeForModel(BufferedImage image, int targetSize) {
        int origW = image.getWidth();
        int origH = image.getHeight();

        float scale = (float) targetSize / Math.max(origW, origH);
        int scaledW = roundToMultipleOf(Math.round(origW * scale), PATCH_MULTIPLE);
        int scaledH = roundToMultipleOf(Math.round(origH * scale), PATCH_MULTIPLE);

        BufferedImage result = new BufferedImage(scaledW, scaledH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = result.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.drawImage(image, 0, 0, scaledW, scaledH, null);
        g.dispose();
        return result;
    }

    static int roundToMultipleOf(int value, int multiple) {
        return Math.max(multiple, ((value + multiple - 1) / multiple) * multiple);
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

                data[y * w + x] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                data[h * w + y * w + x] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                data[2 * h * w + y * w + x] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
            }
        }

        return Tensor.fromFloats(data, new long[]{1, 3, h, w});
    }

    // --- Post-processing ---

    private static DepthMap decodeDepth(Map<String, Tensor> outputs, int origW, int origH) {
        Tensor depth = findDepthOutput(outputs);
        float[][] raw = toDepthPlane(depth);
        float[][] resampled = resample(raw, origW, origH);
        return new DepthMap(resampled, origW, origH);
    }

    /**
     * Reduces the model output to a single {@code [height][width]} plane.
     *
     * <p>Exports disagree on rank: some emit {@code [1, H, W]}, others {@code [1, 1, H, W]}.
     * Squeezing all size-1 dimensions handles both without guessing.
     */
    static float[][] toDepthPlane(Tensor depth) {
        Tensor squeezed = depth.squeeze();
        long[] shape = squeezed.shape();
        if (shape.length != 2) {
            throw new InferenceException(
                    "Expected a 2D depth map after squeezing but got shape with "
                            + shape.length + " dimensions");
        }
        return squeezed.toFloats2D();
    }

    static Tensor findDepthOutput(Map<String, Tensor> outputs) {
        Tensor named = outputs.get("predicted_depth");
        if (named != null) {
            return named;
        }
        return outputs.values().iterator().next();
    }

    /**
     * Bilinearly resamples a depth plane to the target dimensions.
     *
     * <p>Depth is predicted at the model's working resolution; callers expect it in the
     * coordinates of the image they passed in.
     */
    static float[][] resample(float[][] src, int targetW, int targetH) {
        int srcH = src.length;
        int srcW = src[0].length;
        if (srcH == targetH && srcW == targetW) {
            return src;
        }

        float[][] dst = new float[targetH][targetW];
        float scaleX = srcW > 1 ? (float) (srcW - 1) / Math.max(1, targetW - 1) : 0f;
        float scaleY = srcH > 1 ? (float) (srcH - 1) / Math.max(1, targetH - 1) : 0f;

        for (int y = 0; y < targetH; y++) {
            float sy = y * scaleY;
            int y0 = (int) sy;
            int y1 = Math.min(y0 + 1, srcH - 1);
            float wy = sy - y0;

            for (int x = 0; x < targetW; x++) {
                float sx = x * scaleX;
                int x0 = (int) sx;
                int x1 = Math.min(x0 + 1, srcW - 1);
                float wx = sx - x0;

                float top = src[y0][x0] * (1 - wx) + src[y0][x1] * wx;
                float bottom = src[y1][x0] * (1 - wx) + src[y1][x1] * wx;
                dst[y][x] = top * (1 - wy) + bottom * wy;
            }
        }
        return dst;
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
        private String inputName;
        private int targetSize = DEFAULT_TARGET_SIZE;

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

        /**
         * Sets the target size for the longer edge before inference.
         *
         * <p>Larger values recover more detail at higher cost. Rounded up to a
         * multiple of 14, the model's patch size. Defaults to 518.
         *
         * @param targetSize longer-edge size in pixels
         * @return this builder
         */
        public Builder targetSize(int targetSize) {
            this.targetSize = targetSize;
            return this;
        }

        public DepthAnythingEstimator build() {
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
            return new DepthAnythingEstimator(session, inputName, targetSize);
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
