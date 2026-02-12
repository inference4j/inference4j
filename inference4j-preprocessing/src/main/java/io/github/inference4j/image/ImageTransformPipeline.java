package io.github.inference4j.image;

import io.github.inference4j.Tensor;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Composes image transforms and produces a normalized float tensor.
 *
 * <p>Supports both NCHW (channel-first, PyTorch) and NHWC (channel-last, TensorFlow) layouts.
 *
 * <p>Preset usage:
 * <pre>{@code
 * ImageTransformPipeline pipeline = ImageTransformPipeline.imagenet(224);
 * Tensor tensor = pipeline.transform(image);
 * }</pre>
 *
 * <p>Custom pipeline:
 * <pre>{@code
 * ImageTransformPipeline pipeline = ImageTransformPipeline.builder()
 *     .resize(256, 256)
 *     .centerCrop(224, 224)
 *     .mean(new float[]{0.485f, 0.456f, 0.406f})
 *     .std(new float[]{0.229f, 0.224f, 0.225f})
 *     .layout(ImageLayout.NHWC)
 *     .build();
 * }</pre>
 */
public class ImageTransformPipeline {

    private static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};

    private final List<ImageTransform> transforms;
    private final float[] mean;
    private final float[] std;
    private final ImageLayout layout;

    private ImageTransformPipeline(List<ImageTransform> transforms, float[] mean, float[] std,
                                   ImageLayout layout) {
        this.transforms = List.copyOf(transforms);
        this.mean = mean.clone();
        this.std = std.clone();
        this.layout = layout;
    }

    /**
     * Standard ImageNet preprocessing: resize(256) → centerCrop(size) → normalize(ImageNet mean/std).
     * Outputs NCHW layout.
     */
    public static ImageTransformPipeline imagenet(int size) {
        return builder()
                .resize(256, 256)
                .centerCrop(size, size)
                .mean(IMAGENET_MEAN)
                .std(IMAGENET_STD)
                .build();
    }

    /**
     * EfficientNet preprocessing: resize → centerCrop → normalize((pixel-127)/128).
     * Outputs NHWC layout.
     */
    public static ImageTransformPipeline efficientnet(int size) {
        float m = 127f / 255f;
        float s = 128f / 255f;
        return builder()
                .resize(size, size)
                .centerCrop(size, size)
                .mean(new float[]{m, m, m})
                .std(new float[]{s, s, s})
                .layout(ImageLayout.NHWC)
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Applies all image transforms, then converts to a normalized float tensor.
     *
     * @return Tensor with shape [1, 3, H, W] (NCHW) or [1, H, W, 3] (NHWC)
     */
    public Tensor transform(BufferedImage image) {
        for (ImageTransform t : transforms) {
            image = t.apply(image);
        }
        return toTensor(image);
    }

    private Tensor toTensor(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        float[] data = new float[3 * h * w];

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = image.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255f;
                float g = ((rgb >> 8) & 0xFF) / 255f;
                float b = (rgb & 0xFF) / 255f;

                float nr = (r - mean[0]) / std[0];
                float ng = (g - mean[1]) / std[1];
                float nb = (b - mean[2]) / std[2];

                if (layout == ImageLayout.NHWC) {
                    int idx = (y * w + x) * 3;
                    data[idx]     = nr;
                    data[idx + 1] = ng;
                    data[idx + 2] = nb;
                } else {
                    data[0 * h * w + y * w + x] = nr;
                    data[1 * h * w + y * w + x] = ng;
                    data[2 * h * w + y * w + x] = nb;
                }
            }
        }

        long[] shape = layout == ImageLayout.NHWC
                ? new long[]{1, h, w, 3}
                : new long[]{1, 3, h, w};
        return Tensor.fromFloats(data, shape);
    }

    public static class Builder {
        private final List<ImageTransform> transforms = new ArrayList<>();
        private float[] mean = IMAGENET_MEAN;
        private float[] std = IMAGENET_STD;
        private ImageLayout layout = ImageLayout.NCHW;

        public Builder resize(int width, int height) {
            transforms.add(new ResizeTransform(width, height));
            return this;
        }

        public Builder centerCrop(int width, int height) {
            transforms.add(new CenterCropTransform(width, height));
            return this;
        }

        public Builder addTransform(ImageTransform transform) {
            transforms.add(transform);
            return this;
        }

        public Builder mean(float[] mean) {
            this.mean = mean;
            return this;
        }

        public Builder std(float[] std) {
            this.std = std;
            return this;
        }

        public Builder layout(ImageLayout layout) {
            this.layout = layout;
            return this;
        }

        public ImageTransformPipeline build() {
            return new ImageTransformPipeline(transforms, mean, std, layout);
        }
    }
}
