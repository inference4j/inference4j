package io.github.inference4j.core;

import io.github.inference4j.core.exception.TensorConversionException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorTest {

    @Test
    void fromFloats_createsWithCorrectShapeAndType() {
        Tensor tensor = Tensor.fromFloats(new float[]{1.0f, 2.0f, 3.0f}, new long[]{1, 3});
        assertArrayEquals(new long[]{1, 3}, tensor.shape());
        assertEquals(TensorType.FLOAT, tensor.type());
    }

    @Test
    void fromLongs_createsWithCorrectShapeAndType() {
        Tensor tensor = Tensor.fromLongs(new long[]{10, 20, 30}, new long[]{3});
        assertArrayEquals(new long[]{3}, tensor.shape());
        assertEquals(TensorType.LONG, tensor.type());
    }

    @Test
    void fromFloats_throwsOnShapeMismatch() {
        TensorConversionException ex = assertThrows(TensorConversionException.class, () ->
                Tensor.fromFloats(new float[]{1.0f, 2.0f}, new long[]{1, 3}));
        assertTrue(ex.getMessage().contains("3"));
        assertTrue(ex.getMessage().contains("2"));
    }

    @Test
    void toFloats_returnsDataCopy() {
        float[] original = {1.0f, 2.0f, 3.0f};
        Tensor tensor = Tensor.fromFloats(original, new long[]{3});
        float[] result = tensor.toFloats();
        assertArrayEquals(original, result);
        result[0] = 999f;
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f}, tensor.toFloats(), "should return defensive copy");
    }

    @Test
    void toLongs_returnsDataCopy() {
        long[] original = {10, 20, 30};
        Tensor tensor = Tensor.fromLongs(original, new long[]{3});
        long[] result = tensor.toLongs();
        assertArrayEquals(original, result);
        result[0] = 999;
        assertArrayEquals(new long[]{10, 20, 30}, tensor.toLongs(), "should return defensive copy");
    }

    @Test
    void toFloats_throwsOnTypeMismatch() {
        Tensor tensor = Tensor.fromLongs(new long[]{1, 2, 3}, new long[]{3});
        TensorConversionException ex = assertThrows(TensorConversionException.class, tensor::toFloats);
        assertTrue(ex.getMessage().contains("LONG"));
        assertTrue(ex.getMessage().contains("FLOAT"));
    }

    @Test
    void toLongs_throwsOnTypeMismatch() {
        Tensor tensor = Tensor.fromFloats(new float[]{1.0f}, new long[]{1});
        TensorConversionException ex = assertThrows(TensorConversionException.class, tensor::toLongs);
        assertTrue(ex.getMessage().contains("FLOAT"));
        assertTrue(ex.getMessage().contains("LONG"));
    }

    @Test
    void toFloats2D_reshapesCorrectly() {
        float[] flat = {1f, 2f, 3f, 4f, 5f, 6f};
        Tensor tensor = Tensor.fromFloats(flat, new long[]{2, 3});
        float[][] result = tensor.toFloats2D();
        assertArrayEquals(new float[]{1f, 2f, 3f}, result[0]);
        assertArrayEquals(new float[]{4f, 5f, 6f}, result[1]);
    }

    @Test
    void toFloats2D_throwsOnNon2DShape() {
        Tensor tensor = Tensor.fromFloats(new float[]{1f, 2f, 3f}, new long[]{3});
        TensorConversionException ex = assertThrows(TensorConversionException.class, tensor::toFloats2D);
        assertTrue(ex.getMessage().contains("2D"));
    }

    @Test
    void shape_returnsDefensiveCopy() {
        long[] shape = {2, 3};
        Tensor tensor = Tensor.fromFloats(new float[6], shape);
        tensor.shape()[0] = 999;
        assertArrayEquals(new long[]{2, 3}, tensor.shape());
    }

    @Test
    void fromFloats_makesDefensiveCopyOfInput() {
        float[] data = {1f, 2f, 3f};
        Tensor tensor = Tensor.fromFloats(data, new long[]{3});
        data[0] = 999f;
        assertArrayEquals(new float[]{1f, 2f, 3f}, tensor.toFloats());
    }
}
