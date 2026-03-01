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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.model.ModelSource;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class CraftTextDetectorTest {

    // --- Connected Component Tests ---

    @Test
    void connectedComponents_singleComponent() {
        // 3x3 grid with a single 2x2 block
        boolean[] binary = {
                true, true, false,
                true, true, false,
                false, false, false
        };

        int[] labels = CraftTextDetector.connectedComponents(binary, 3, 3);

        assertThat(labels[0]).isEqualTo(1); // (0,0)
        assertThat(labels[1]).isEqualTo(1); // (1,0)
        assertThat(labels[3]).isEqualTo(1); // (0,1)
        assertThat(labels[4]).isEqualTo(1); // (1,1)
        assertThat(labels[2]).isEqualTo(0); // background
        assertThat(labels[5]).isEqualTo(0); // background
        assertThat(labels[6]).isEqualTo(0); // background
        assertThat(labels[7]).isEqualTo(0); // background
        assertThat(labels[8]).isEqualTo(0); // background
    }

    @Test
    void connectedComponents_multipleComponents() {
        // 5x3 grid with two separate blocks
        boolean[] binary = {
                true, true, false, true, true,
                true, true, false, true, true,
                false, false, false, false, false
        };

        int[] labels = CraftTextDetector.connectedComponents(binary, 5, 3);

        // First component
        int label1 = labels[0];
        assertThat(label1).isGreaterThan(0);
        assertThat(labels[1]).isEqualTo(label1);
        assertThat(labels[5]).isEqualTo(label1);
        assertThat(labels[6]).isEqualTo(label1);

        // Second component — different label
        int label2 = labels[3];
        assertThat(label2).isGreaterThan(0);
        assertThat(label2).isNotEqualTo(label1);
        assertThat(labels[4]).isEqualTo(label2);
        assertThat(labels[8]).isEqualTo(label2);
        assertThat(labels[9]).isEqualTo(label2);
    }

    @Test
    void connectedComponents_noForeground() {
        boolean[] binary = {false, false, false, false};

        int[] labels = CraftTextDetector.connectedComponents(binary, 2, 2);

        for (int label : labels) {
            assertThat(label).isEqualTo(0);
        }
    }

    @Test
    void connectedComponents_allForeground() {
        boolean[] binary = {true, true, true, true};

        int[] labels = CraftTextDetector.connectedComponents(binary, 2, 2);

        assertThat(labels[0]).isEqualTo(1);
        assertThat(labels[1]).isEqualTo(1);
        assertThat(labels[2]).isEqualTo(1);
        assertThat(labels[3]).isEqualTo(1);
    }

    @Test
    void connectedComponents_diagonalNotConnected() {
        // 4-connectivity: diagonal pixels should NOT be connected
        boolean[] binary = {
                true, false,
                false, true
        };

        int[] labels = CraftTextDetector.connectedComponents(binary, 2, 2);

        assertThat(labels[0]).isGreaterThan(0);
        assertThat(labels[3]).isGreaterThan(0);
        assertThat(labels[3]).as("Diagonal pixels should be separate components with 4-connectivity").isNotEqualTo(labels[0]);
    }

    @Test
    void connectedComponents_lShaped() {
        // L-shaped component
        boolean[] binary = {
                true, false, false,
                true, false, false,
                true, true, true
        };

        int[] labels = CraftTextDetector.connectedComponents(binary, 3, 3);

        int label = labels[0];
        assertThat(label).isGreaterThan(0);
        assertThat(labels[3]).isEqualTo(label); // (0,1)
        assertThat(labels[6]).isEqualTo(label); // (0,2)
        assertThat(labels[7]).isEqualTo(label); // (1,2)
        assertThat(labels[8]).isEqualTo(label); // (2,2)
    }

    // --- Post-processing Tests ---

    @Test
    void postProcess_singleTextRegion() {
        // 4x4 heatmap with a 2x2 hot region at top-left
        float[] region = createHeatmap(4, 4, 0f, new HotSpot(0, 0, 2, 2, 0.9f));
        float[] affinity = createHeatmap(4, 4, 0f, new HotSpot(0, 0, 2, 2, 0.1f));

        // scale=1, origW=8, origH=8 (heatmap is half resolution)
        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 1, 1.0f, 8, 8);

        assertThat(results).hasSize(1);
        TextRegion r = results.get(0);
        assertThat(r.confidence()).isCloseTo(0.9f, within(1e-3f));
        // Heatmap coords [0,0]-[2,2), scaled: x1=0*2/1=0, y1=0*2/1=0, x2=2*2/1=4, y2=2*2/1=4
        assertThat(r.box().x1()).isCloseTo(0f, within(1e-1f));
        assertThat(r.box().y1()).isCloseTo(0f, within(1e-1f));
        assertThat(r.box().x2()).isCloseTo(4f, within(1e-1f));
        assertThat(r.box().y2()).isCloseTo(4f, within(1e-1f));
    }

    @Test
    void postProcess_filtersSmallComponents() {
        // Single-pixel hot spot (area=1, below minArea=2)
        float[] region = createHeatmap(4, 4, 0f, new HotSpot(1, 1, 1, 1, 0.9f));
        float[] affinity = createHeatmap(4, 4, 0f, new HotSpot(1, 1, 1, 1, 0.1f));

        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 2, 1.0f, 8, 8);

        assertThat(results).as("Single-pixel component should be filtered by minArea=2").isEmpty();
    }

    @Test
    void postProcess_filtersLowConfidence() {
        // Region with low mean region score (0.3 < textThreshold=0.7)
        float[] region = createHeatmap(4, 4, 0f, new HotSpot(0, 0, 2, 2, 0.3f));
        float[] affinity = createHeatmap(4, 4, 0f, new HotSpot(0, 0, 2, 2, 0.2f));

        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 1, 1.0f, 8, 8);

        assertThat(results).as("Low-confidence component should be filtered").isEmpty();
    }

    @Test
    void postProcess_scalesCoordinates() {
        // Heatmap 4x4, scale=0.5 means original image was 2x bigger before rescale
        float[] region = createHeatmap(4, 4, 0f, new HotSpot(1, 1, 2, 2, 0.9f));
        float[] affinity = createHeatmap(4, 4, 0f, new HotSpot(1, 1, 2, 2, 0.1f));

        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 1, 0.5f, 16, 16);

        assertThat(results).hasSize(1);
        BoundingBox box = results.get(0).box();
        // minX=1, minY=1, maxX=2, maxY=2
        // x1 = 1 * 2 / 0.5 = 4, y1 = 1 * 2 / 0.5 = 4
        // x2 = (2+1) * 2 / 0.5 = 12, y2 = (2+1) * 2 / 0.5 = 12
        assertThat(box.x1()).isCloseTo(4f, within(1e-1f));
        assertThat(box.y1()).isCloseTo(4f, within(1e-1f));
        assertThat(box.x2()).isCloseTo(12f, within(1e-1f));
        assertThat(box.y2()).isCloseTo(12f, within(1e-1f));
    }

    @Test
    void postProcess_clipsToImageBounds() {
        // Hot spot at the edge of heatmap, scaled coordinates would exceed image bounds
        float[] region = createHeatmap(4, 4, 0f, new HotSpot(2, 2, 2, 2, 0.9f));
        float[] affinity = createHeatmap(4, 4, 0f, new HotSpot(2, 2, 2, 2, 0.1f));

        // Small original image (4x4) so scaled coords will exceed
        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 1, 1.0f, 4, 4);

        assertThat(results).hasSize(1);
        BoundingBox box = results.get(0).box();
        assertThat(box.x2()).as("x2 should be clipped to origW").isLessThanOrEqualTo(4);
        assertThat(box.y2()).as("y2 should be clipped to origH").isLessThanOrEqualTo(4);
    }

    @Test
    void postProcess_sortedByConfidenceDescending() {
        // Two separate hot spots with different scores
        float[] region = createHeatmap(8, 4, 0f,
                new HotSpot(0, 0, 2, 2, 0.8f),
                new HotSpot(5, 0, 2, 2, 0.95f));
        float[] affinity = createHeatmap(8, 4, 0f,
                new HotSpot(0, 0, 2, 2, 0.1f),
                new HotSpot(5, 0, 2, 2, 0.1f));

        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 8,
                0.7f, 0.4f, 1, 1.0f, 16, 8);

        assertThat(results).hasSize(2);
        assertThat(results.get(0).confidence())
                .as("Results should be sorted by confidence descending")
                .isGreaterThanOrEqualTo(results.get(1).confidence());
        assertThat(results.get(0).confidence()).isCloseTo(0.95f, within(1e-3f));
        assertThat(results.get(1).confidence()).isCloseTo(0.8f, within(1e-3f));
    }

    @Test
    void postProcess_emptyWhenNothingAboveThreshold() {
        // All zeros
        float[] region = new float[16];
        float[] affinity = new float[16];

        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 1, 1.0f, 8, 8);

        assertThat(results).isEmpty();
    }

    // --- Preprocessing Tests ---

    @Test
    void imageToTensor_appliesImageNetNormalization() {
        // Create 2x2 image with known pixel values
        BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
        // Set pixel (0,0) to pure red (255, 0, 0)
        image.setRGB(0, 0, 0xFF0000);

        Tensor tensor = CraftTextDetector.imageToTensor(image);

        float[] data = tensor.toFloats();
        long[] shape = tensor.shape();
        assertThat(shape).isEqualTo(new long[]{1, 3, 2, 2});

        // Red channel at (0,0): (1.0 - 0.485) / 0.229 ~ 2.2489
        float expectedR = (1.0f - 0.485f) / 0.229f;
        assertThat(data[0]).isCloseTo(expectedR, within(1e-3f));

        // Green channel at (0,0): (0.0 - 0.456) / 0.224 ~ -2.0357
        float expectedG = (0.0f - 0.456f) / 0.224f;
        assertThat(data[4]).isCloseTo(expectedG, within(1e-3f));

        // Blue channel at (0,0): (0.0 - 0.406) / 0.225 ~ -1.8044
        float expectedB = (0.0f - 0.406f) / 0.225f;
        assertThat(data[8]).isCloseTo(expectedB, within(1e-3f));
    }

    @Test
    void resizeForCraft_preservesAspectRatio() {
        BufferedImage image = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);

        CraftTextDetector.ResizeResult result = CraftTextDetector.resizeForCraft(image, 320);

        // Long side should be near 320, short side proportional
        assertThat(result.image().getWidth()).as("Width should be near targetSize").isLessThanOrEqualTo(320 + 32);
        assertThat(result.image().getHeight()).as("Height should be near targetSize").isLessThanOrEqualTo(320 + 32);
        assertThat(result.scale()).isGreaterThan(0);
    }

    @Test
    void resizeForCraft_dimensionsMultipleOf32() {
        BufferedImage image = new BufferedImage(700, 500, BufferedImage.TYPE_INT_RGB);

        CraftTextDetector.ResizeResult result = CraftTextDetector.resizeForCraft(image, 640);

        assertThat(result.image().getWidth() % 32).as("Width should be multiple of 32").isEqualTo(0);
        assertThat(result.image().getHeight() % 32).as("Height should be multiple of 32").isEqualTo(0);
    }

    @Test
    void roundToMultipleOf32_roundsCorrectly() {
        assertThat(CraftTextDetector.roundToMultipleOf32(1)).isEqualTo(32);
        assertThat(CraftTextDetector.roundToMultipleOf32(16)).isEqualTo(32);
        assertThat(CraftTextDetector.roundToMultipleOf32(17)).isEqualTo(32);
        assertThat(CraftTextDetector.roundToMultipleOf32(32)).isEqualTo(32);
        assertThat(CraftTextDetector.roundToMultipleOf32(33)).isEqualTo(64);
        assertThat(CraftTextDetector.roundToMultipleOf32(48)).isEqualTo(64);
        assertThat(CraftTextDetector.roundToMultipleOf32(49)).isEqualTo(64);
        assertThat(CraftTextDetector.roundToMultipleOf32(64)).isEqualTo(64);
        assertThat(CraftTextDetector.roundToMultipleOf32(640)).isEqualTo(640);
    }

    // --- Integration Test ---

    @Test
    void detect_bufferedImage_endToEnd() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input"));

        // Create synthetic heatmap output [1, H/2, W/2, 2]
        // Input is 32x32, so heatmap is 16x16
        int heatmapH = 16;
        int heatmapW = 16;
        float[] outputData = new float[heatmapH * heatmapW * 2];

        // Place a hot spot at (4,4)-(8,8) in heatmap with high region and low affinity
        for (int y = 4; y < 8; y++) {
            for (int x = 4; x < 8; x++) {
                int idx = y * heatmapW * 2 + x * 2;
                outputData[idx] = 0.9f;     // region score
                outputData[idx + 1] = 0.1f; // affinity score
            }
        }

        when(session.run(any())).thenReturn(
                Map.of("output", Tensor.fromFloats(outputData,
                        new long[]{1, heatmapH, heatmapW, 2})));

        CraftTextDetector craft = CraftTextDetector.builder()
                .session(session)
                .inputName("input")
                .targetSize(32)
                .textThreshold(0.7f)
                .lowTextThreshold(0.4f)
                .minComponentArea(1)
                .build();

        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        List<TextRegion> results = craft.detect(image);

        assertThat(results).as("Should detect at least one text region").isNotEmpty();
        assertThat(results.get(0).confidence()).isCloseTo(0.9f, within(1e-2f));
    }

    @Test
    void detect_twoOutputTensors_usesScoreMap() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input"));

        int heatmapH = 16;
        int heatmapW = 16;

        // score_map: [1, H/2, W/2, 2] — region + affinity channels
        float[] scoreMapData = new float[heatmapH * heatmapW * 2];
        for (int y = 4; y < 8; y++) {
            for (int x = 4; x < 8; x++) {
                int idx = y * heatmapW * 2 + x * 2;
                scoreMapData[idx] = 0.85f;     // region score
                scoreMapData[idx + 1] = 0.15f; // affinity score
            }
        }

        // feature_map: [1, 32, H/2, W/2] — intermediate features (should be ignored)
        float[] featureMapData = new float[32 * heatmapH * heatmapW];

        // Use LinkedHashMap to preserve insertion order, matching real ONNX output
        Map<String, Tensor> outputMap = new LinkedHashMap<>();
        outputMap.put("score_map", Tensor.fromFloats(scoreMapData,
                new long[]{1, heatmapH, heatmapW, 2}));
        outputMap.put("feature_map", Tensor.fromFloats(featureMapData,
                new long[]{1, 32, heatmapH, heatmapW}));

        when(session.run(any())).thenReturn(outputMap);

        CraftTextDetector craft = CraftTextDetector.builder()
                .session(session)
                .inputName("input")
                .targetSize(32)
                .minComponentArea(1)
                .build();

        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        List<TextRegion> results = craft.detect(image);

        assertThat(results).as("Should detect text from score_map output").isNotEmpty();
        assertThat(results.get(0).confidence()).isCloseTo(0.85f, within(1e-2f));
    }

    // --- Builder Tests ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                CraftTextDetector.builder()
                        .inputName("input")
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_inputNameDefaultsFromSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input"));

        CraftTextDetector model = CraftTextDetector.builder()
                .session(session)
                .build();

        assertThat(model).isNotNull();
        verify(session).inputNames();
    }

    // --- Close Tests ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);

        CraftTextDetector model = CraftTextDetector.builder()
                .session(session)
                .inputName("input")
                .build();

        model.close();

        verify(session).close();
    }

    // --- Test Helpers ---

    record HotSpot(int x, int y, int width, int height, float value) {
    }

    /**
     * Creates a heatmap with a uniform background and one or more hot spots.
     */
    private static float[] createHeatmap(int width, int height, float background, HotSpot... hotSpots) {
        float[] heatmap = new float[width * height];
        for (int i = 0; i < heatmap.length; i++) {
            heatmap[i] = background;
        }
        for (HotSpot spot : hotSpots) {
            for (int sy = spot.y; sy < spot.y + spot.height && sy < height; sy++) {
                for (int sx = spot.x; sx < spot.x + spot.width && sx < width; sx++) {
                    heatmap[sy * width + sx] = spot.value;
                }
            }
        }
        return heatmap;
    }
}
