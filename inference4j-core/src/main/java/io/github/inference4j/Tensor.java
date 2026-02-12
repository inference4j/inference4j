package io.github.inference4j;

import io.github.inference4j.exception.TensorConversionException;
import java.util.Arrays;

/**
 * An immutable, typed multi-dimensional array of numeric data.
 *
 * <p>Tensors are the data exchange format between Java code and ONNX Runtime.
 * They hold a flat data array and a shape that defines the logical dimensions.
 * Data is defensively copied on creation and retrieval.
 *
 * <p>Create tensors via the static factory methods:
 * <pre>{@code
 * // 1D tensor with 3 elements
 * Tensor t1 = Tensor.fromFloats(new float[]{1f, 2f, 3f}, new long[]{3});
 *
 * // 2D tensor (batch of 2, embedding dim 4)
 * Tensor t2 = Tensor.fromFloats(data, new long[]{2, 4});
 * }</pre>
 *
 * @see TensorType
 * @see InferenceSession#run(java.util.Map)
 */
public class Tensor {

    private final Object data;
    private final long[] shape;
    private final TensorType type;

    private Tensor(Object data, long[] shape, TensorType type) {
        this.data = data;
        this.shape = shape.clone();
        this.type = type;
    }

    /**
     * Creates a float tensor with the given data and shape.
     *
     * @param data  flat array of float values
     * @param shape the tensor dimensions (e.g., {@code {1, 3, 224, 224}})
     * @return a new float tensor
     * @throws TensorConversionException if data length does not match the shape
     */
    public static Tensor fromFloats(float[] data, long[] shape) {
        validateShape(data.length, shape);
        return new Tensor(data.clone(), shape, TensorType.FLOAT);
    }

    /**
     * Creates a long tensor with the given data and shape.
     *
     * @param data  flat array of long values
     * @param shape the tensor dimensions (e.g., {@code {1, 128}})
     * @return a new long tensor
     * @throws TensorConversionException if data length does not match the shape
     */
    public static Tensor fromLongs(long[] data, long[] shape) {
        validateShape(data.length, shape);
        return new Tensor(data.clone(), shape, TensorType.LONG);
    }

    /** Returns a copy of this tensor's shape. */
    public long[] shape() {
        return shape.clone();
    }

    /** Returns this tensor's element type. */
    public TensorType type() {
        return type;
    }

    /**
     * Returns this tensor's data as a flat float array.
     *
     * @return a copy of the underlying float data
     * @throws TensorConversionException if this is not a {@link TensorType#FLOAT} tensor
     */
    public float[] toFloats() {
        if (type != TensorType.FLOAT) {
            throw new TensorConversionException(
                    "Cannot convert " + type + " tensor to FLOAT");
        }
        return ((float[]) data).clone();
    }

    /**
     * Returns this tensor's data as a flat long array.
     *
     * @return a copy of the underlying long data
     * @throws TensorConversionException if this is not a {@link TensorType#LONG} tensor
     */
    public long[] toLongs() {
        if (type != TensorType.LONG) {
            throw new TensorConversionException(
                    "Cannot convert " + type + " tensor to LONG");
        }
        return ((long[]) data).clone();
    }

    /**
     * Returns this tensor's data as a 2D float array, reshaped according to the tensor's shape.
     *
     * @return a {@code [rows][cols]} array
     * @throws TensorConversionException if this is not a {@link TensorType#FLOAT} tensor
     *                                   or the shape is not 2-dimensional
     */
    public float[][] toFloats2D() {
        if (type != TensorType.FLOAT) {
            throw new TensorConversionException(
                    "Cannot convert " + type + " tensor to FLOAT");
        }
        if (shape.length != 2) {
            throw new TensorConversionException(
                    "Cannot reshape to 2D: tensor has " + shape.length + " dimensions, expected 2D shape");
        }
        float[] flat = (float[]) data;
        int rows = (int) shape[0];
        int cols = (int) shape[1];
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, result[i], 0, cols);
        }
        return result;
    }

    // Package-private: used by InferenceSession for zero-copy tensor creation
    Object rawData() {
        return data;
    }

    private static void validateShape(int dataLength, long[] shape) {
        long expected = 1;
        for (long dim : shape) {
            expected *= dim;
        }
        if (dataLength != expected) {
            throw new TensorConversionException(
                    "Data length " + dataLength + " does not match shape " +
                            Arrays.toString(shape) + " (expected " + expected + " elements)");
        }
    }
}
