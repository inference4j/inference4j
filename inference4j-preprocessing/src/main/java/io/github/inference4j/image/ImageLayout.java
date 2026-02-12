package io.github.inference4j.image;

/**
 * Tensor layout for image data.
 *
 * <ul>
 *   <li>{@link #NCHW} — [batch, channels, height, width] (PyTorch-origin models)</li>
 *   <li>{@link #NHWC} — [batch, height, width, channels] (TensorFlow-origin models)</li>
 * </ul>
 */
public enum ImageLayout {
    NCHW,
    NHWC;

    /**
     * Detects image layout from a 4D tensor shape by locating the channels dimension (1 or 3).
     *
     * @param shape 4D tensor shape [batch, ?, ?, ?]
     * @return detected layout
     * @throws IllegalArgumentException if shape is not 4D
     */
    public static ImageLayout detect(long[] shape) {
        if (shape.length != 4) {
            throw new IllegalArgumentException(
                    "Expected 4D tensor shape, got " + shape.length + "D");
        }
        if (shape[1] == 3 || shape[1] == 1) return NCHW;
        if (shape[3] == 3 || shape[3] == 1) return NHWC;
        // Ambiguous (e.g. [1,3,3,3]) — default to NCHW (PyTorch convention)
        return NCHW;
    }

    /**
     * Extracts the spatial image size (height) from the shape.
     */
    public int imageSize(long[] shape) {
        return (int) (this == NCHW ? shape[2] : shape[1]);
    }
}
