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
     * Creates a string tensor with the given data and shape.
     *
     * @param data  flat array of string values
     * @param shape the tensor dimensions (e.g., {@code {1, 3}})
     * @return a new string tensor
     * @throws TensorConversionException if data length does not match the shape
     */
    public static Tensor fromStrings(String[] data, long[] shape) {
        validateShape(data.length, shape);
        return new Tensor(data.clone(), shape, TensorType.STRING);
    }

    /**
     * Creates a float16 tensor with the given data and shape.
     *
     * <p>Each {@code short} value holds a raw IEEE 754 half-precision float.
     * This format is used internally for efficient KV cache pass-through in
     * FP16 models. Call {@link #toFloats()} to convert to float32.
     *
     * @param data  flat array of raw FP16 values (as shorts)
     * @param shape the tensor dimensions
     * @return a new float16 tensor
     * @throws TensorConversionException if data length does not match the shape
     */
    public static Tensor fromFloat16(short[] data, long[] shape) {
        validateShape(data.length, shape);
        return new Tensor(data.clone(), shape, TensorType.FLOAT16);
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
     * <p>{@link TensorType#FLOAT16} tensors are automatically converted to float32.
     *
     * @return a copy of the underlying float data (or converted from float16)
     * @throws TensorConversionException if this tensor's type cannot be converted to float
     */
    public float[] toFloats() {
        if (type == TensorType.FLOAT) {
            return ((float[]) data).clone();
        }
        if (type == TensorType.FLOAT16) {
            short[] fp16 = (short[]) data;
            float[] result = new float[fp16.length];
            for (int i = 0; i < fp16.length; i++) {
                result[i] = fp16ToFloat32(fp16[i]);
            }
            return result;
        }
        throw new TensorConversionException(
                "Cannot convert " + type + " tensor to FLOAT");
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

    /**
     * Returns this tensor's data as a flat string array.
     *
     * @return a copy of the underlying string data
     * @throws TensorConversionException if this is not a {@link TensorType#STRING} tensor
     */
    public String[] toStrings() {
        if (type != TensorType.STRING) {
            throw new TensorConversionException(
                    "Cannot convert " + type + " tensor to STRING");
        }
        return ((String[]) data).clone();
    }

    /**
     * Extracts a single index along the given axis, reducing dimensionality by one.
     * Supports negative indexing: {@code -1} is the last element, {@code -2} second-to-last, etc.
     *
     * <p>This is the tensor equivalent of Python's array indexing along an axis.
     * For a 3D tensor with shape {@code [1, 5, 768]}:
     * <pre>{@code
     * tensor.slice(0, 0)   // shape [5, 768]  — remove batch dimension
     * tensor.slice(1, -1)  // shape [1, 768]  — last position along axis 1
     * }</pre>
     *
     * <p>Chaining works naturally for multi-step extraction:
     * <pre>{@code
     * // logits [1, promptLength, vocabSize] → last position's logits [vocabSize]
     * float[] lastLogits = logits.slice(0, 0).slice(0, -1).toFloats();
     * }</pre>
     *
     * @param axis  the dimension to slice along (0-indexed)
     * @param index the index to extract (negative values count from the end)
     * @return a new tensor with one fewer dimension
     * @throws TensorConversionException if axis or index is out of range
     */
    public Tensor slice(int axis, int index) {
        if (axis < 0 || axis >= shape.length) {
            throw new TensorConversionException(
                    "Axis " + axis + " out of range for " + shape.length + "D tensor");
        }

        int axisSize = (int) shape[axis];

        if (index < 0) {
            index += axisSize;
        }
        if (index < 0 || index >= axisSize) {
            throw new TensorConversionException(
                    "Index " + index + " out of range for axis " + axis + " with size " + axisSize);
        }

        // New shape: original with this axis removed
        long[] newShape = new long[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                newShape[j++] = shape[i];
            }
        }

        // innerSize = product of all dimensions after the sliced axis
        // This is the "stride" — how many elements one step along this axis covers
        int innerSize = 1;
        for (int i = axis + 1; i < shape.length; i++) {
            innerSize *= (int) shape[i];
        }

        // outerSize = product of all dimensions before the sliced axis
        // This is how many independent "blocks" we need to copy
        int outerSize = 1;
        for (int i = 0; i < axis; i++) {
            outerSize *= (int) shape[i];
        }

        return sliceCopy(outerSize, axisSize, innerSize, index, newShape);
    }

    /**
     * Removes all dimensions of size 1.
     *
     * <p>For a tensor with shape {@code [1, 5, 1, 768]}, returns a tensor
     * with shape {@code [5, 768]}. The underlying data is unchanged.
     *
     * @return a new tensor with size-1 dimensions removed, or this tensor if none exist
     */
    public Tensor squeeze() {
        long[] newShape = Arrays.stream(shape).filter(d -> d != 1).toArray();
        if (newShape.length == shape.length) {
            return this;
        }
        if (newShape.length == 0) {
            newShape = new long[]{1};
        }
        return new Tensor(data, newShape, type);
    }

    /**
     * Removes a specific dimension of size 1.
     *
     * <p>For a tensor with shape {@code [1, 5, 768]}, {@code squeeze(0)} returns
     * a tensor with shape {@code [5, 768]}. Throws if the dimension is not size 1.
     *
     * @param axis the dimension to remove (must be size 1)
     * @return a new tensor with the specified dimension removed
     * @throws TensorConversionException if axis is out of range or dimension is not size 1
     */
    public Tensor squeeze(int axis) {
        if (axis < 0 || axis >= shape.length) {
            throw new TensorConversionException(
                    "Axis " + axis + " out of range for " + shape.length + "D tensor");
        }
        if (shape[axis] != 1) {
            throw new TensorConversionException(
                    "Cannot squeeze axis " + axis + " with size " + shape[axis] + " (must be 1)");
        }
        long[] newShape = new long[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                newShape[j++] = shape[i];
            }
        }
        return new Tensor(data, newShape, type);
    }

    private Tensor sliceCopy(int outerSize, int axisSize, int innerSize,
                             int index, long[] newShape) {
        // For each outer block, copy innerSize elements from the selected index
        // Source layout: [outer][axisIndex][inner] — contiguous in memory
        // We pick one axisIndex and flatten the rest
        return switch (type) {
            case FLOAT -> {
                float[] src = (float[]) data;
                float[] dst = new float[outerSize * innerSize];
                for (int outer = 0; outer < outerSize; outer++) {
                    System.arraycopy(
                            src, outer * axisSize * innerSize + index * innerSize,
                            dst, outer * innerSize,
                            innerSize);
                }
                yield new Tensor(dst, newShape, TensorType.FLOAT);
            }
            case LONG -> {
                long[] src = (long[]) data;
                long[] dst = new long[outerSize * innerSize];
                for (int outer = 0; outer < outerSize; outer++) {
                    System.arraycopy(
                            src, outer * axisSize * innerSize + index * innerSize,
                            dst, outer * innerSize,
                            innerSize);
                }
                yield new Tensor(dst, newShape, TensorType.LONG);
            }
            case FLOAT16 -> {
                short[] src = (short[]) data;
                short[] dst = new short[outerSize * innerSize];
                for (int outer = 0; outer < outerSize; outer++) {
                    System.arraycopy(
                            src, outer * axisSize * innerSize + index * innerSize,
                            dst, outer * innerSize,
                            innerSize);
                }
                yield new Tensor(dst, newShape, TensorType.FLOAT16);
            }
            case STRING -> {
                String[] src = (String[]) data;
                String[] dst = new String[outerSize * innerSize];
                for (int outer = 0; outer < outerSize; outer++) {
                    System.arraycopy(
                            src, outer * axisSize * innerSize + index * innerSize,
                            dst, outer * innerSize,
                            innerSize);
                }
                yield new Tensor(dst, newShape, TensorType.STRING);
            }
            default -> throw new TensorConversionException(
                    "Unsupported tensor type for slice: " + type);
        };
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

    /**
     * Converts an IEEE 754 half-precision (FP16) value to single-precision (FP32).
     */
    static float fp16ToFloat32(short fp16) {
        int bits = fp16 & 0xFFFF;
        int sign = (bits >> 15) & 1;
        int exp = (bits >> 10) & 0x1F;
        int mantissa = bits & 0x3FF;

        if (exp == 0) {
            if (mantissa == 0) {
                return Float.intBitsToFloat(sign << 31);
            }
            // Subnormal: normalize
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            exp++;
            mantissa &= 0x3FF;
        } else if (exp == 31) {
            // Infinity or NaN
            return Float.intBitsToFloat(
                    (sign << 31) | 0x7F800000 | (mantissa << 13));
        }

        return Float.intBitsToFloat(
                (sign << 31) | ((exp - 15 + 127) << 23) | (mantissa << 13));
    }
}
