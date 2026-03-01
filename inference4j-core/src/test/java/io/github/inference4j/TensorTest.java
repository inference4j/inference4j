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
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class TensorTest {

    @Test
    void fromFloats_createsWithCorrectShapeAndType() {
        Tensor tensor = Tensor.fromFloats(new float[]{1.0f, 2.0f, 3.0f}, new long[]{1, 3});
        assertThat(tensor.shape()).isEqualTo(new long[]{1, 3});
        assertThat(tensor.type()).isEqualTo(TensorType.FLOAT);
    }

    @Test
    void fromLongs_createsWithCorrectShapeAndType() {
        Tensor tensor = Tensor.fromLongs(new long[]{10, 20, 30}, new long[]{3});
        assertThat(tensor.shape()).isEqualTo(new long[]{3});
        assertThat(tensor.type()).isEqualTo(TensorType.LONG);
    }

    @Test
    void fromFloats_throwsOnShapeMismatch() {
        assertThatThrownBy(() ->
                Tensor.fromFloats(new float[]{1.0f, 2.0f}, new long[]{1, 3}))
                .isInstanceOf(TensorConversionException.class)
                .satisfies(ex -> {
                    assertThat(ex.getMessage()).contains("3");
                    assertThat(ex.getMessage()).contains("2");
                });
    }

    @Test
    void toFloats_returnsDataCopy() {
        float[] original = {1.0f, 2.0f, 3.0f};
        Tensor tensor = Tensor.fromFloats(original, new long[]{3});
        float[] result = tensor.toFloats();
        assertThat(result).isEqualTo(original);
        result[0] = 999f;
        assertThat(tensor.toFloats()).as("should return defensive copy").isEqualTo(new float[]{1.0f, 2.0f, 3.0f});
    }

    @Test
    void toLongs_returnsDataCopy() {
        long[] original = {10, 20, 30};
        Tensor tensor = Tensor.fromLongs(original, new long[]{3});
        long[] result = tensor.toLongs();
        assertThat(result).isEqualTo(original);
        result[0] = 999;
        assertThat(tensor.toLongs()).as("should return defensive copy").isEqualTo(new long[]{10, 20, 30});
    }

    @Test
    void toFloats_throwsOnTypeMismatch() {
        Tensor tensor = Tensor.fromLongs(new long[]{1, 2, 3}, new long[]{3});
        assertThatThrownBy(tensor::toFloats)
                .isInstanceOf(TensorConversionException.class)
                .satisfies(ex -> {
                    assertThat(ex.getMessage()).contains("LONG");
                    assertThat(ex.getMessage()).contains("FLOAT");
                });
    }

    @Test
    void toLongs_throwsOnTypeMismatch() {
        Tensor tensor = Tensor.fromFloats(new float[]{1.0f}, new long[]{1});
        assertThatThrownBy(tensor::toLongs)
                .isInstanceOf(TensorConversionException.class)
                .satisfies(ex -> {
                    assertThat(ex.getMessage()).contains("FLOAT");
                    assertThat(ex.getMessage()).contains("LONG");
                });
    }

    @Test
    void toFloats2D_reshapesCorrectly() {
        float[] flat = {1f, 2f, 3f, 4f, 5f, 6f};
        Tensor tensor = Tensor.fromFloats(flat, new long[]{2, 3});
        float[][] result = tensor.toFloats2D();
        assertThat(result[0]).isEqualTo(new float[]{1f, 2f, 3f});
        assertThat(result[1]).isEqualTo(new float[]{4f, 5f, 6f});
    }

    @Test
    void toFloats2D_throwsOnNon2DShape() {
        Tensor tensor = Tensor.fromFloats(new float[]{1f, 2f, 3f}, new long[]{3});
        assertThatThrownBy(tensor::toFloats2D)
                .isInstanceOf(TensorConversionException.class)
                .satisfies(ex -> assertThat(ex.getMessage()).contains("2D"));
    }

    @Test
    void shape_returnsDefensiveCopy() {
        long[] shape = {2, 3};
        Tensor tensor = Tensor.fromFloats(new float[6], shape);
        tensor.shape()[0] = 999;
        assertThat(tensor.shape()).isEqualTo(new long[]{2, 3});
    }

    @Test
    void fromFloats_makesDefensiveCopyOfInput() {
        float[] data = {1f, 2f, 3f};
        Tensor tensor = Tensor.fromFloats(data, new long[]{3});
        data[0] = 999f;
        assertThat(tensor.toFloats()).isEqualTo(new float[]{1f, 2f, 3f});
    }

    @Test
    void fromStrings_createsWithCorrectShapeAndType() {
        Tensor tensor = Tensor.fromStrings(new String[]{"hello", "world"}, new long[]{1, 2});
        assertThat(tensor.shape()).isEqualTo(new long[]{1, 2});
        assertThat(tensor.type()).isEqualTo(TensorType.STRING);
    }

    @Test
    void fromStrings_throwsOnShapeMismatch() {
        assertThatThrownBy(() ->
                Tensor.fromStrings(new String[]{"a", "b"}, new long[]{1, 3}))
                .isInstanceOf(TensorConversionException.class)
                .satisfies(ex -> {
                    assertThat(ex.getMessage()).contains("3");
                    assertThat(ex.getMessage()).contains("2");
                });
    }

    @Test
    void toStrings_returnsDataCopy() {
        String[] original = {"hello", "world", "test"};
        Tensor tensor = Tensor.fromStrings(original, new long[]{3});
        String[] result = tensor.toStrings();
        assertThat(result).isEqualTo(original);
        result[0] = "modified";
        assertThat(tensor.toStrings()).as("should return defensive copy").isEqualTo(new String[]{"hello", "world", "test"});
    }

    @Test
    void toStrings_throwsOnTypeMismatch() {
        Tensor tensor = Tensor.fromFloats(new float[]{1.0f}, new long[]{1});
        assertThatThrownBy(tensor::toStrings)
                .isInstanceOf(TensorConversionException.class)
                .satisfies(ex -> {
                    assertThat(ex.getMessage()).contains("FLOAT");
                    assertThat(ex.getMessage()).contains("STRING");
                });
    }

    @Test
    void fromStrings_makesDefensiveCopyOfInput() {
        String[] data = {"hello", "world"};
        Tensor tensor = Tensor.fromStrings(data, new long[]{2});
        data[0] = "modified";
        assertThat(tensor.toStrings()).isEqualTo(new String[]{"hello", "world"});
    }
}
