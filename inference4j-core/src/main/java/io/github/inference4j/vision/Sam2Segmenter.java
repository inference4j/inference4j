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

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.HuggingFaceModelSource;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.session.SessionConfigurer;

/**
 * Promptable segmentation with SAM 2.1 (Segment Anything Model 2.1).
 *
 * <p>Click a point or draw a box and SAM returns the exact outline of the object there, for any
 * kind of object: it was never trained on a fixed list of classes. Use it for background removal
 * and cut-outs, for annotation tools, or after an {@link ObjectDetector} to turn each box into a
 * precise mask.
 *
 * <h2>Tested model</h2>
 * <p><a href="https://huggingface.co/inference4j/sam2.1-hiera-tiny">inference4j/sam2.1-hiera-tiny</a>
 * — the Tiny variant (~39M parameters), two graphs: an image encoder and a prompt decoder.
 *
 * <h2>How it runs</h2>
 * <ul>
 *   <li>{@link #encode} resizes the image to 1024&times;1024 (bilinear, aspect ratio <em>not</em>
 *       preserved, as SAM 2 expects), normalizes with ImageNet mean/std, and runs the encoder.
 *       This is the slow step.</li>
 *   <li>{@link #segment} scales the prompt into that 1024&times;1024 space and runs the decoder,
 *       which proposes three 256&times;256 masks with a predicted IoU each. Each mask is
 *       upsampled to 1024&times;1024, then to the original image size, and thresholded at 0.</li>
 * </ul>
 *
 * <h2>Quick start</h2>
 * <pre>{@code
 * try (var sam = Sam2Segmenter.builder().build()) {
 *     ImageFeatures features = sam.encode(image);
 *     SegmentationMask mask = sam.segment(features, Prompt.point(320, 240));
 *     ImageAnnotator.save(mask.applyTo(image), Path.of("cutout.png"));
 * }
 * }</pre>
 *
 * <h2>With an object detector</h2>
 * <pre>{@code
 * ImageFeatures features = sam.encode(image);
 * for (Detection d : detector.detect(image)) {
 *     SegmentationMask mask = sam.segment(features, Prompt.box(d.box()));
 * }
 * }</pre>
 *
 * @see PromptableSegmenter
 * @see SegmentationMask
 */
public class Sam2Segmenter implements PromptableSegmenter {

    private static final String DEFAULT_MODEL_ID = "inference4j/sam2.1-hiera-tiny";
    private static final String ENCODER_FILE = "vision_encoder.onnx";
    private static final String DECODER_FILE = "prompt_encoder_mask_decoder.onnx";
    private static final List<String> MODEL_FILES = List.of(
            ENCODER_FILE, ENCODER_FILE + "_data",
            DECODER_FILE, DECODER_FILE + "_data");

    /** SAM 2 works on a fixed square input; images are stretched to it, not letterboxed. */
    static final int INPUT_SIZE = 1024;

    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    private static final String PIXEL_VALUES = "pixel_values";
    private static final List<String> EMBEDDINGS = List.of(
            "image_embeddings.0", "image_embeddings.1", "image_embeddings.2");
    private static final String INPUT_POINTS = "input_points";
    private static final String INPUT_LABELS = "input_labels";
    private static final String INPUT_BOXES = "input_boxes";
    private static final String IOU_SCORES = "iou_scores";
    private static final String PRED_MASKS = "pred_masks";

    private final InferenceSession encoder;
    private final InferenceSession decoder;

    private Sam2Segmenter(InferenceSession encoder, InferenceSession decoder) {
        this.encoder = encoder;
        this.decoder = decoder;
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    public ImageFeatures encode(BufferedImage image) {
        Map<String, Tensor> outputs = encoder.run(Map.of(PIXEL_VALUES, imageToTensor(image)));
        Map<String, Tensor> embeddings = new HashMap<>();
        for (String name : EMBEDDINGS) {
            Tensor tensor = outputs.get(name);
            if (tensor == null) {
                throw new InferenceException("Encoder did not produce '" + name + "'");
            }
            embeddings.put(name, tensor);
        }
        return new ImageFeatures(embeddings, image.getWidth(), image.getHeight());
    }

    @Override
    public ImageFeatures encode(Path imagePath) {
        return encode(loadImage(imagePath));
    }

    @Override
    public SegmentationMask segment(ImageFeatures features, Prompt prompt) {
        Proposals proposals = decode(features, prompt);
        int best = 0;
        for (int i = 1; i < proposals.scores().length; i++) {
            if (proposals.scores()[i] > proposals.scores()[best]) {
                best = i;
            }
        }
        return proposals.toMask(best, features.width(), features.height());
    }

    @Override
    public List<SegmentationMask> segmentCandidates(ImageFeatures features, Prompt prompt) {
        Proposals proposals = decode(features, prompt);
        List<SegmentationMask> masks = new ArrayList<>();
        for (int i = 0; i < proposals.scores().length; i++) {
            masks.add(proposals.toMask(i, features.width(), features.height()));
        }
        masks.sort(Comparator.comparingDouble(SegmentationMask::score).reversed());
        return masks;
    }

    @Override
    public void close() {
        try {
            encoder.close();
        } finally {
            decoder.close();
        }
    }

    // --- Preprocessing ---

    /** Stretches the image to 1024&times;1024 and normalizes it into an NCHW tensor. */
    static Tensor imageToTensor(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        int[] rgb = image.getRGB(0, 0, w, h, null, 0, w);
        int[] resized = PilImages.resizeBilinear(rgb, w, h, INPUT_SIZE, INPUT_SIZE);

        int plane = INPUT_SIZE * INPUT_SIZE;
        float[] data = new float[3 * plane];
        for (int i = 0; i < plane; i++) {
            int p = resized[i];
            data[i] = (((p >> 16) & 0xFF) / 255f - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
            data[plane + i] = (((p >> 8) & 0xFF) / 255f - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
            data[2 * plane + i] = ((p & 0xFF) / 255f - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
        }
        return Tensor.fromFloats(data, new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
    }

    /**
     * Builds the decoder's prompt inputs, scaling each coordinate from the original image into
     * the 1024&times;1024 model space. Absent clicks or box become empty tensors, which the
     * decoder accepts.
     */
    static Map<String, Tensor> promptTensors(Prompt prompt, int width, int height) {
        float sx = (float) INPUT_SIZE / width;
        float sy = (float) INPUT_SIZE / height;

        List<Prompt.Click> clicks = prompt.clicks();
        float[] points = new float[clicks.size() * 2];
        long[] labels = new long[clicks.size()];
        for (int i = 0; i < clicks.size(); i++) {
            Prompt.Click click = clicks.get(i);
            points[2 * i] = click.point().x() * sx;
            points[2 * i + 1] = click.point().y() * sy;
            labels[i] = click.foreground() ? 1 : 0;
        }

        float[] boxes = prompt.box()
                .map(b -> new float[]{b.x1() * sx, b.y1() * sy, b.x2() * sx, b.y2() * sy})
                .orElse(new float[0]);

        return Map.of(
                INPUT_POINTS, Tensor.fromFloats(points, new long[]{1, 1, clicks.size(), 2}),
                INPUT_LABELS, Tensor.fromLongs(labels, new long[]{1, 1, clicks.size()}),
                INPUT_BOXES, Tensor.fromFloats(boxes, new long[]{1, boxes.length / 4, 4}));
    }

    // --- Decoding ---

    private Proposals decode(ImageFeatures features, Prompt prompt) {
        Map<String, Tensor> inputs = new HashMap<>(features.tensors());
        inputs.putAll(promptTensors(prompt, features.width(), features.height()));
        Map<String, Tensor> outputs = decoder.run(inputs);

        Tensor scores = outputs.get(IOU_SCORES);
        Tensor masks = outputs.get(PRED_MASKS);
        if (scores == null || masks == null) {
            throw new InferenceException("Decoder did not produce '" + IOU_SCORES + "' and '" + PRED_MASKS + "'");
        }
        long[] shape = masks.shape();
        if (shape.length != 5) {
            throw new InferenceException("Expected pred_masks of rank 5 but got rank " + shape.length);
        }
        return new Proposals(scores.toFloats(), masks.toFloats(), (int) shape[3], (int) shape[4]);
    }

    /** The decoder's candidate masks for one prompt, as low-resolution logits. */
    private record Proposals(float[] scores, float[] logits, int maskHeight, int maskWidth) {

        SegmentationMask toMask(int index, int width, int height) {
            int size = maskHeight * maskWidth;
            float[] plane = new float[size];
            System.arraycopy(logits, index * size, plane, 0, size);
            return Sam2Segmenter.toMask(plane, maskWidth, maskHeight, width, height, scores[index]);
        }
    }

    /**
     * Upsamples mask logits to the original image size and thresholds them at 0.
     *
     * <p>Follows SAM 2's post-processing: first to the 1024&times;1024 model input, then to the
     * original size, both bilinear with {@code align_corners=False}.
     */
    static SegmentationMask toMask(float[] logits, int srcWidth, int srcHeight,
                                   int width, int height, float score) {
        float[] model = resizeBilinear(logits, srcWidth, srcHeight, INPUT_SIZE, INPUT_SIZE);
        float[] full = resizeBilinear(model, INPUT_SIZE, INPUT_SIZE, width, height);
        boolean[][] mask = new boolean[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                mask[y][x] = full[y * width + x] > 0f;
            }
        }
        return new SegmentationMask(mask, width, height, score);
    }

    /**
     * Bilinear resize of a single-channel plane, matching PyTorch's
     * {@code F.interpolate(mode="bilinear", align_corners=False)} without antialiasing.
     */
    static float[] resizeBilinear(float[] src, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
        if (srcWidth == dstWidth && srcHeight == dstHeight) {
            return src;
        }
        int[] x0 = new int[dstWidth];
        int[] x1 = new int[dstWidth];
        float[] wx = new float[dstWidth];
        sourceCoordinates(srcWidth, dstWidth, x0, x1, wx);
        int[] y0 = new int[dstHeight];
        int[] y1 = new int[dstHeight];
        float[] wy = new float[dstHeight];
        sourceCoordinates(srcHeight, dstHeight, y0, y1, wy);

        float[] dst = new float[dstWidth * dstHeight];
        for (int y = 0; y < dstHeight; y++) {
            int top = y0[y] * srcWidth;
            int bottom = y1[y] * srcWidth;
            for (int x = 0; x < dstWidth; x++) {
                float t = src[top + x0[x]] * (1 - wx[x]) + src[top + x1[x]] * wx[x];
                float b = src[bottom + x0[x]] * (1 - wx[x]) + src[bottom + x1[x]] * wx[x];
                dst[y * dstWidth + x] = t * (1 - wy[y]) + b * wy[y];
            }
        }
        return dst;
    }

    /** For each output index, the two source indices it blends and the weight of the second. */
    private static void sourceCoordinates(int srcSize, int dstSize, int[] i0, int[] i1, float[] weight) {
        float scale = (float) srcSize / dstSize;
        for (int i = 0; i < dstSize; i++) {
            float s = Math.max((i + 0.5f) * scale - 0.5f, 0f);
            int lo = Math.min((int) s, srcSize - 1);
            i0[i] = lo;
            i1[i] = Math.min(lo + 1, srcSize - 1);
            weight[i] = s - lo;
        }
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
        private InferenceSession encoderSession;
        private InferenceSession decoderSession;
        private ModelSource modelSource;
        private String modelId;
        private SessionConfigurer sessionConfigurer;

        Builder encoderSession(InferenceSession session) {
            this.encoderSession = session;
            return this;
        }

        Builder decoderSession(InferenceSession session) {
            this.decoderSession = session;
            return this;
        }

        /**
         * Configures both ONNX Runtime sessions, e.g. to add an execution provider.
         *
         * @param sessionConfigurer applied to the encoder and the decoder session
         * @return this builder
         */
        public Builder sessionOptions(SessionConfigurer sessionConfigurer) {
            this.sessionConfigurer = sessionConfigurer;
            return this;
        }

        public Builder modelSource(ModelSource modelSource) {
            this.modelSource = modelSource;
            return this;
        }

        /**
         * Sets the model to load. The directory must hold {@code vision_encoder.onnx} and
         * {@code prompt_encoder_mask_decoder.onnx}, each with its {@code .onnx_data} weights file.
         *
         * @param modelId the model identifier; defaults to {@code inference4j/sam2.1-hiera-tiny}
         * @return this builder
         */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Sam2Segmenter build() {
            if (encoderSession == null || decoderSession == null) {
                ModelSource source = modelSource != null
                        ? modelSource : HuggingFaceModelSource.defaultInstance();
                String id = modelId != null ? modelId : DEFAULT_MODEL_ID;
                loadFromDirectory(source.resolve(id, MODEL_FILES));
            }
            return new Sam2Segmenter(encoderSession, decoderSession);
        }

        private void loadFromDirectory(Path dir) {
            if (!Files.isDirectory(dir)) {
                throw new ModelSourceException("Model directory not found: " + dir);
            }
            Path encoderPath = requireFile(dir, ENCODER_FILE);
            Path decoderPath = requireFile(dir, DECODER_FILE);

            this.encoderSession = open(encoderPath);
            try {
                this.decoderSession = open(decoderPath);
            } catch (RuntimeException e) {
                encoderSession.close();
                throw e;
            }
        }

        private InferenceSession open(Path path) {
            return sessionConfigurer != null
                    ? InferenceSession.create(path, sessionConfigurer)
                    : InferenceSession.create(path);
        }

        private static Path requireFile(Path dir, String name) {
            Path path = dir.resolve(name);
            if (!Files.exists(path)) {
                throw new ModelSourceException("Model file not found: " + path);
            }
            return path;
        }
    }
}
