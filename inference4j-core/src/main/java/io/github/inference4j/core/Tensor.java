package io.github.inference4j.core;

import io.github.inference4j.core.exception.TensorConversionException;
import java.util.Arrays;

public class Tensor {

    private final Object data;
    private final long[] shape;
    private final TensorType type;

    private Tensor(Object data, long[] shape, TensorType type) {
        this.data = data;
        this.shape = shape.clone();
        this.type = type;
    }

    public static Tensor fromFloats(float[] data, long[] shape) {
        validateShape(data.length, shape);
        return new Tensor(data.clone(), shape, TensorType.FLOAT);
    }

    public static Tensor fromLongs(long[] data, long[] shape) {
        validateShape(data.length, shape);
        return new Tensor(data.clone(), shape, TensorType.LONG);
    }

    public long[] shape() {
        return shape.clone();
    }

    public TensorType type() {
        return type;
    }

    public float[] toFloats() {
        if (type != TensorType.FLOAT) {
            throw new TensorConversionException(
                    "Cannot convert " + type + " tensor to FLOAT");
        }
        return ((float[]) data).clone();
    }

    public long[] toLongs() {
        if (type != TensorType.LONG) {
            throw new TensorConversionException(
                    "Cannot convert " + type + " tensor to LONG");
        }
        return ((long[]) data).clone();
    }

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
