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

import static org.junit.jupiter.api.Assertions.*;
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

        assertEquals(1, labels[0]); // (0,0)
        assertEquals(1, labels[1]); // (1,0)
        assertEquals(1, labels[3]); // (0,1)
        assertEquals(1, labels[4]); // (1,1)
        assertEquals(0, labels[2]); // background
        assertEquals(0, labels[5]); // background
        assertEquals(0, labels[6]); // background
        assertEquals(0, labels[7]); // background
        assertEquals(0, labels[8]); // background
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
        assertTrue(label1 > 0);
        assertEquals(label1, labels[1]);
        assertEquals(label1, labels[5]);
        assertEquals(label1, labels[6]);

        // Second component — different label
        int label2 = labels[3];
        assertTrue(label2 > 0);
        assertNotEquals(label1, label2);
        assertEquals(label2, labels[4]);
        assertEquals(label2, labels[8]);
        assertEquals(label2, labels[9]);
    }

    @Test
    void connectedComponents_noForeground() {
        boolean[] binary = {false, false, false, false};

        int[] labels = CraftTextDetector.connectedComponents(binary, 2, 2);

        for (int label : labels) {
            assertEquals(0, label);
        }
    }

    @Test
    void connectedComponents_allForeground() {
        boolean[] binary = {true, true, true, true};

        int[] labels = CraftTextDetector.connectedComponents(binary, 2, 2);

        assertEquals(1, labels[0]);
        assertEquals(1, labels[1]);
        assertEquals(1, labels[2]);
        assertEquals(1, labels[3]);
    }

    @Test
    void connectedComponents_diagonalNotConnected() {
        // 4-connectivity: diagonal pixels should NOT be connected
        boolean[] binary = {
                true, false,
                false, true
        };

        int[] labels = CraftTextDetector.connectedComponents(binary, 2, 2);

        assertTrue(labels[0] > 0);
        assertTrue(labels[3] > 0);
        assertNotEquals(labels[0], labels[3], "Diagonal pixels should be separate components with 4-connectivity");
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
        assertTrue(label > 0);
        assertEquals(label, labels[3]); // (0,1)
        assertEquals(label, labels[6]); // (0,2)
        assertEquals(label, labels[7]); // (1,2)
        assertEquals(label, labels[8]); // (2,2)
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

        assertEquals(1, results.size());
        TextRegion r = results.get(0);
        assertEquals(0.9f, r.confidence(), 1e-3f);
        // Heatmap coords [0,0]-[2,2), scaled: x1=0*2/1=0, y1=0*2/1=0, x2=2*2/1=4, y2=2*2/1=4
        assertEquals(0f, r.box().x1(), 1e-1f);
        assertEquals(0f, r.box().y1(), 1e-1f);
        assertEquals(4f, r.box().x2(), 1e-1f);
        assertEquals(4f, r.box().y2(), 1e-1f);
    }

    @Test
    void postProcess_filtersSmallComponents() {
        // Single-pixel hot spot (area=1, below minArea=2)
        float[] region = createHeatmap(4, 4, 0f, new HotSpot(1, 1, 1, 1, 0.9f));
        float[] affinity = createHeatmap(4, 4, 0f, new HotSpot(1, 1, 1, 1, 0.1f));

        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 2, 1.0f, 8, 8);

        assertTrue(results.isEmpty(), "Single-pixel component should be filtered by minArea=2");
    }

    @Test
    void postProcess_filtersLowConfidence() {
        // Region with low mean region score (0.3 < textThreshold=0.7)
        float[] region = createHeatmap(4, 4, 0f, new HotSpot(0, 0, 2, 2, 0.3f));
        float[] affinity = createHeatmap(4, 4, 0f, new HotSpot(0, 0, 2, 2, 0.2f));

        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 1, 1.0f, 8, 8);

        assertTrue(results.isEmpty(), "Low-confidence component should be filtered");
    }

    @Test
    void postProcess_scalesCoordinates() {
        // Heatmap 4x4, scale=0.5 means original image was 2x bigger before rescale
        float[] region = createHeatmap(4, 4, 0f, new HotSpot(1, 1, 2, 2, 0.9f));
        float[] affinity = createHeatmap(4, 4, 0f, new HotSpot(1, 1, 2, 2, 0.1f));

        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 1, 0.5f, 16, 16);

        assertEquals(1, results.size());
        BoundingBox box = results.get(0).box();
        // minX=1, minY=1, maxX=2, maxY=2
        // x1 = 1 * 2 / 0.5 = 4, y1 = 1 * 2 / 0.5 = 4
        // x2 = (2+1) * 2 / 0.5 = 12, y2 = (2+1) * 2 / 0.5 = 12
        assertEquals(4f, box.x1(), 1e-1f);
        assertEquals(4f, box.y1(), 1e-1f);
        assertEquals(12f, box.x2(), 1e-1f);
        assertEquals(12f, box.y2(), 1e-1f);
    }

    @Test
    void postProcess_clipsToImageBounds() {
        // Hot spot at the edge of heatmap, scaled coordinates would exceed image bounds
        float[] region = createHeatmap(4, 4, 0f, new HotSpot(2, 2, 2, 2, 0.9f));
        float[] affinity = createHeatmap(4, 4, 0f, new HotSpot(2, 2, 2, 2, 0.1f));

        // Small original image (4x4) so scaled coords will exceed
        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 1, 1.0f, 4, 4);

        assertEquals(1, results.size());
        BoundingBox box = results.get(0).box();
        assertTrue(box.x2() <= 4, "x2 should be clipped to origW");
        assertTrue(box.y2() <= 4, "y2 should be clipped to origH");
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

        assertEquals(2, results.size());
        assertTrue(results.get(0).confidence() >= results.get(1).confidence(),
                "Results should be sorted by confidence descending");
        assertEquals(0.95f, results.get(0).confidence(), 1e-3f);
        assertEquals(0.8f, results.get(1).confidence(), 1e-3f);
    }

    @Test
    void postProcess_emptyWhenNothingAboveThreshold() {
        // All zeros
        float[] region = new float[16];
        float[] affinity = new float[16];

        List<TextRegion> results = CraftTextDetector.postProcess(region, affinity, 4, 4,
                0.7f, 0.4f, 1, 1.0f, 8, 8);

        assertTrue(results.isEmpty());
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
        assertArrayEquals(new long[]{1, 3, 2, 2}, shape);

        // Red channel at (0,0): (1.0 - 0.485) / 0.229 ~ 2.2489
        float expectedR = (1.0f - 0.485f) / 0.229f;
        assertEquals(expectedR, data[0], 1e-3f);

        // Green channel at (0,0): (0.0 - 0.456) / 0.224 ~ -2.0357
        float expectedG = (0.0f - 0.456f) / 0.224f;
        assertEquals(expectedG, data[4], 1e-3f);

        // Blue channel at (0,0): (0.0 - 0.406) / 0.225 ~ -1.8044
        float expectedB = (0.0f - 0.406f) / 0.225f;
        assertEquals(expectedB, data[8], 1e-3f);
    }

    @Test
    void resizeForCraft_preservesAspectRatio() {
        BufferedImage image = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);

        CraftTextDetector.ResizeResult result = CraftTextDetector.resizeForCraft(image, 320);

        // Long side should be near 320, short side proportional
        assertTrue(result.image().getWidth() <= 320 + 32, "Width should be near targetSize");
        assertTrue(result.image().getHeight() <= 320 + 32, "Height should be near targetSize");
        assertTrue(result.scale() > 0);
    }

    @Test
    void resizeForCraft_dimensionsMultipleOf32() {
        BufferedImage image = new BufferedImage(700, 500, BufferedImage.TYPE_INT_RGB);

        CraftTextDetector.ResizeResult result = CraftTextDetector.resizeForCraft(image, 640);

        assertEquals(0, result.image().getWidth() % 32, "Width should be multiple of 32");
        assertEquals(0, result.image().getHeight() % 32, "Height should be multiple of 32");
    }

    @Test
    void roundToMultipleOf32_roundsCorrectly() {
        assertEquals(32, CraftTextDetector.roundToMultipleOf32(1));
        assertEquals(32, CraftTextDetector.roundToMultipleOf32(16));
        assertEquals(32, CraftTextDetector.roundToMultipleOf32(17));
        assertEquals(32, CraftTextDetector.roundToMultipleOf32(32));
        assertEquals(64, CraftTextDetector.roundToMultipleOf32(33));
        assertEquals(64, CraftTextDetector.roundToMultipleOf32(48));
        assertEquals(64, CraftTextDetector.roundToMultipleOf32(49));
        assertEquals(64, CraftTextDetector.roundToMultipleOf32(64));
        assertEquals(640, CraftTextDetector.roundToMultipleOf32(640));
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

        assertFalse(results.isEmpty(), "Should detect at least one text region");
        assertEquals(0.9f, results.get(0).confidence(), 1e-2f);
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

        assertFalse(results.isEmpty(), "Should detect text from score_map output");
        assertEquals(0.85f, results.get(0).confidence(), 1e-2f);
    }

    // --- Builder Tests ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThrows(ModelSourceException.class, () ->
                CraftTextDetector.builder()
                        .inputName("input")
                        .modelSource(badSource)
                        .build());
    }

    @Test
    void builder_inputNameDefaultsFromSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("input"));

        CraftTextDetector model = CraftTextDetector.builder()
                .session(session)
                .build();

        assertNotNull(model);
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
