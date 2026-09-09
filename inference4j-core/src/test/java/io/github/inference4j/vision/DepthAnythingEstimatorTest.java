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
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.ModelSource;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class DepthAnythingEstimatorTest {

    // --- Resize Tests ---

    @Test
    void resizeForModel_roundsBothDimensionsToPatchMultiple() {
        BufferedImage image = new BufferedImage(300, 200, BufferedImage.TYPE_INT_RGB);
        BufferedImage resized = DepthAnythingEstimator.resizeForModel(image, 518);

        assertThat(resized.getWidth() % 14).as("Width must be a multiple of 14").isZero();
        assertThat(resized.getHeight() % 14).as("Height must be a multiple of 14").isZero();
    }

    @Test
    void resizeForModel_preservesAspectRatioApproximately() {
        BufferedImage image = new BufferedImage(400, 200, BufferedImage.TYPE_INT_RGB);
        BufferedImage resized = DepthAnythingEstimator.resizeForModel(image, 518);

        float originalRatio = 400f / 200f;
        float resizedRatio = (float) resized.getWidth() / resized.getHeight();
        // Rounding to a multiple of 14 perturbs the ratio slightly
        assertThat(resizedRatio).isCloseTo(originalRatio, within(0.15f));
    }

    @Test
    void resizeForModel_scalesLongerEdgeToTarget() {
        BufferedImage image = new BufferedImage(1000, 500, BufferedImage.TYPE_INT_RGB);
        BufferedImage resized = DepthAnythingEstimator.resizeForModel(image, 518);

        assertThat(resized.getWidth()).isBetween(504, 532);
    }

    @Test
    void roundToMultipleOf_roundsUpAndEnforcesMinimum() {
        assertThat(DepthAnythingEstimator.roundToMultipleOf(1, 14)).isEqualTo(14);
        assertThat(DepthAnythingEstimator.roundToMultipleOf(14, 14)).isEqualTo(14);
        assertThat(DepthAnythingEstimator.roundToMultipleOf(15, 14)).isEqualTo(28);
        assertThat(DepthAnythingEstimator.roundToMultipleOf(0, 14)).isEqualTo(14);
    }

    // --- Preprocessing Tests ---

    @Test
    void imageToTensor_appliesImageNetNormalizationInNchw() {
        BufferedImage image = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
        image.setRGB(0, 0, 0xFF8000);

        Tensor tensor = DepthAnythingEstimator.imageToTensor(image);
        float[] data = tensor.toFloats();

        assertThat(tensor.shape()).isEqualTo(new long[]{1, 3, 1, 1});
        assertThat(data[0]).isCloseTo((1f - 0.485f) / 0.229f, within(1e-4f));
        assertThat(data[1]).isCloseTo((128f / 255f - 0.456f) / 0.224f, within(1e-4f));
        assertThat(data[2]).isCloseTo((0f - 0.406f) / 0.225f, within(1e-4f));
    }

    // --- Depth Plane Tests ---

    @Test
    void toDepthPlane_handlesRank3Output() {
        Tensor depth = Tensor.fromFloats(new float[]{1f, 2f, 3f, 4f}, new long[]{1, 2, 2});
        assertThat(DepthAnythingEstimator.toDepthPlane(depth)[1]).isEqualTo(new float[]{3f, 4f});
    }

    @Test
    void toDepthPlane_handlesRank4Output() {
        // Some exports emit [1, 1, H, W] instead of [1, H, W]
        Tensor depth = Tensor.fromFloats(new float[]{1f, 2f, 3f, 4f}, new long[]{1, 1, 2, 2});
        assertThat(DepthAnythingEstimator.toDepthPlane(depth)[1]).isEqualTo(new float[]{3f, 4f});
    }

    @Test
    void toDepthPlane_throwsWhenNotReducibleTo2D() {
        Tensor depth = Tensor.fromFloats(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f},
                new long[]{2, 2, 2});
        assertThatThrownBy(() -> DepthAnythingEstimator.toDepthPlane(depth))
                .isInstanceOf(InferenceException.class)
                .hasMessageContaining("2D");
    }

    @Test
    void findDepthOutput_prefersNamedOutput() {
        Map<String, Tensor> outputs = new LinkedHashMap<>();
        outputs.put("other", Tensor.fromFloats(new float[]{9f}, new long[]{1, 1, 1}));
        outputs.put("predicted_depth", Tensor.fromFloats(new float[]{1f}, new long[]{1, 1, 1}));

        assertThat(DepthAnythingEstimator.findDepthOutput(outputs).toFloats()).containsExactly(1f);
    }

    @Test
    void findDepthOutput_fallsBackToFirstOutput() {
        Map<String, Tensor> outputs = new LinkedHashMap<>();
        outputs.put("depth", Tensor.fromFloats(new float[]{7f}, new long[]{1, 1, 1}));

        assertThat(DepthAnythingEstimator.findDepthOutput(outputs).toFloats()).containsExactly(7f);
    }

    // --- Resampling Tests ---

    @Test
    void resample_returnsSameArrayWhenDimensionsAlreadyMatch() {
        float[][] src = {{1f, 2f}, {3f, 4f}};
        assertThat(DepthAnythingEstimator.resample(src, 2, 2)).isSameAs(src);
    }

    @Test
    void resample_preservesCornerValues() {
        float[][] src = {{0f, 10f}, {20f, 30f}};
        float[][] result = DepthAnythingEstimator.resample(src, 4, 4);

        assertThat(result[0][0]).isCloseTo(0f, within(1e-4f));
        assertThat(result[0][3]).isCloseTo(10f, within(1e-4f));
        assertThat(result[3][0]).isCloseTo(20f, within(1e-4f));
        assertThat(result[3][3]).isCloseTo(30f, within(1e-4f));
    }

    @Test
    void resample_interpolatesBetweenCorners() {
        float[][] src = {{0f, 10f}, {0f, 10f}};
        float[][] result = DepthAnythingEstimator.resample(src, 3, 3);
        assertThat(result[0][1]).isCloseTo(5f, within(1e-4f));
    }

    @Test
    void resample_producesRequestedDimensions() {
        float[][] result = DepthAnythingEstimator.resample(new float[][]{{1f, 2f}, {3f, 4f}}, 7, 5);
        assertThat(result.length).isEqualTo(5);
        assertThat(result[0].length).isEqualTo(7);
    }

    // --- End-to-End Tests ---

    @Test
    void estimate_returnsDepthMapInOriginalImageDimensions() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.run(any())).thenReturn(Map.of(
                "predicted_depth", Tensor.fromFloats(
                        new float[]{1f, 2f, 3f, 4f}, new long[]{1, 2, 2})));

        try (DepthAnythingEstimator estimator = DepthAnythingEstimator.builder()
                .session(session)
                .inputName("pixel_values")
                .build()) {

            BufferedImage image = new BufferedImage(64, 32, BufferedImage.TYPE_INT_RGB);
            DepthMap depth = estimator.estimate(image);

            assertThat(depth.width()).isEqualTo(64);
            assertThat(depth.height()).isEqualTo(32);
            assertThat(depth.values().length).isEqualTo(32);
            assertThat(depth.values()[0].length).isEqualTo(64);
        }
    }

    @Test
    void estimate_exposesValueRange() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.run(any())).thenReturn(Map.of(
                "predicted_depth", Tensor.fromFloats(
                        new float[]{0f, 5f, 10f, 15f}, new long[]{1, 2, 2})));

        try (DepthAnythingEstimator estimator = DepthAnythingEstimator.builder()
                .session(session)
                .inputName("pixel_values")
                .build()) {

            DepthMap depth = estimator.estimate(new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB));

            assertThat(depth.min()).isCloseTo(0f, within(1e-4f));
            assertThat(depth.max()).isCloseTo(15f, within(1e-4f));
        }
    }

    @Test
    void estimate_producesRenderableImage() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.run(any())).thenReturn(Map.of(
                "predicted_depth", Tensor.fromFloats(
                        new float[]{0f, 1f, 2f, 3f}, new long[]{1, 2, 2})));

        try (DepthAnythingEstimator estimator = DepthAnythingEstimator.builder()
                .session(session)
                .inputName("pixel_values")
                .build()) {

            DepthMap depth = estimator.estimate(new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB));
            BufferedImage rendered = depth.toImage(Colormap.TURBO);

            assertThat(rendered.getWidth()).isEqualTo(16);
            assertThat(rendered.getHeight()).isEqualTo(16);
        }
    }

    // --- Builder Tests ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                DepthAnythingEstimator.builder()
                        .inputName("pixel_values")
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_inputNameDefaultsFromSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("pixel_values"));

        DepthAnythingEstimator model = DepthAnythingEstimator.builder()
                .session(session)
                .build();

        assertThat(model).isNotNull();
        verify(session).inputNames();
    }

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);

        DepthAnythingEstimator model = DepthAnythingEstimator.builder()
                .session(session)
                .inputName("pixel_values")
                .build();

        model.close();

        verify(session).close();
    }
}
