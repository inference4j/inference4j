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

package io.github.inference4j.vision.detection;

import io.github.inference4j.image.Labels;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class YoloV8Test {

    // YOLOv8 output shape [1, numOutputs, numCandidates]
    // where numOutputs = 4 (box) + numClasses
    private static final int NUM_CLASSES = 5;
    private static final int NUM_OUTPUTS = 4 + NUM_CLASSES;

    private static final Labels TEST_LABELS = Labels.of(List.of(
            "person", "bicycle", "car", "motorcycle", "airplane"
    ));

    // No letterbox transform: scale=1, pad=0, image=640x640
    private static final float SCALE = 1f;
    private static final float PAD_X = 0f;
    private static final float PAD_Y = 0f;
    private static final int ORIG_W = 640;
    private static final int ORIG_H = 640;

    /**
     * Creates a synthetic YOLOv8 output tensor with the given candidates.
     * Each candidate is {cx, cy, w, h, class0_score, class1_score, ...}.
     * Output layout: [1, numOutputs, numCandidates] stored row-major.
     */
    private static float[] createOutput(float[]... candidates) {
        int numCandidates = candidates.length;
        float[] output = new float[NUM_OUTPUTS * numCandidates];

        for (int c = 0; c < numCandidates; c++) {
            for (int row = 0; row < NUM_OUTPUTS; row++) {
                output[row * numCandidates + c] = candidates[c][row];
            }
        }
        return output;
    }

    private static long[] shape(int numCandidates) {
        return new long[]{1, NUM_OUTPUTS, numCandidates};
    }

    @Test
    void postProcess_filtersLowConfidenceDetections() {
        float[] output = createOutput(
                // cx, cy, w, h, cls0, cls1, cls2, cls3, cls4
                new float[]{320, 320, 100, 100, 0.9f, 0.1f, 0.0f, 0.0f, 0.0f},  // high conf
                new float[]{100, 100, 50, 50, 0.1f, 0.05f, 0.0f, 0.0f, 0.0f}    // low conf
        );

        List<Detection> results = YoloV8.postProcess(output, shape(2),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        assertEquals("person", results.get(0).label());
    }

    @Test
    void postProcess_correctBoxConversion() {
        // cx=320, cy=240, w=200, h=100 → x1=220, y1=190, x2=420, y2=290
        float[] output = createOutput(
                new float[]{320, 240, 200, 100, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8.postProcess(output, shape(1),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        BoundingBox box = results.get(0).box();
        assertEquals(220f, box.x1(), 1e-3f);
        assertEquals(190f, box.y1(), 1e-3f);
        assertEquals(420f, box.x2(), 1e-3f);
        assertEquals(290f, box.y2(), 1e-3f);
    }

    @Test
    void postProcess_rescalesCoordinatesWithLetterboxParams() {
        // Simulate letterbox: original 1280x720 image scaled to fit 640x640
        // scale = 640/1280 = 0.5, padX = 0, padY = (640-360)/2 = 140
        float scale = 0.5f;
        float padX = 0f;
        float padY = 140f;
        int origW = 1280;
        int origH = 720;

        // Detection at center of letterboxed image: cx=320, cy=320
        // After rescale: x = (320 - 0) / 0.5 = 640, y = (320 - 140) / 0.5 = 360
        float[] output = createOutput(
                new float[]{320, 320, 100, 60, 0.0f, 0.0f, 0.9f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8.postProcess(output, shape(1),
                TEST_LABELS, 0.25f, 0.45f,
                scale, padX, padY, origW, origH);

        assertEquals(1, results.size());
        BoundingBox box = results.get(0).box();
        // cx=320, w=100 → x1=270, x2=370 in letterbox coords
        // rescaled: x1=(270-0)/0.5=540, x2=(370-0)/0.5=740
        assertEquals(540f, box.x1(), 1e-1f);
        assertEquals(740f, box.x2(), 1e-1f);
        // cy=320, h=60 → y1=290, y2=350 in letterbox coords
        // rescaled: y1=(290-140)/0.5=300, y2=(350-140)/0.5=420
        assertEquals(300f, box.y1(), 1e-1f);
        assertEquals(420f, box.y2(), 1e-1f);
        assertEquals("car", results.get(0).label());
    }

    @Test
    void postProcess_nmsSuppressesOverlappingBoxes() {
        // Two highly overlapping detections of the same object
        float[] output = createOutput(
                new float[]{320, 320, 100, 100, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f},
                new float[]{325, 325, 100, 100, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8.postProcess(output, shape(2),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        assertEquals(0.9f, results.get(0).confidence(), 1e-5f);
    }

    @Test
    void postProcess_sortedByConfidenceDescending() {
        // Three non-overlapping detections with different confidences
        float[] output = createOutput(
                new float[]{100, 100, 50, 50, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f},
                new float[]{300, 300, 50, 50, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f},
                new float[]{500, 500, 50, 50, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8.postProcess(output, shape(3),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertEquals(3, results.size());
        assertEquals(0.9f, results.get(0).confidence(), 1e-5f);
        assertEquals(0.7f, results.get(1).confidence(), 1e-5f);
        assertEquals(0.5f, results.get(2).confidence(), 1e-5f);
    }

    @Test
    void postProcess_emptyWhenAllBelowThreshold() {
        float[] output = createOutput(
                new float[]{320, 320, 100, 100, 0.1f, 0.05f, 0.02f, 0.0f, 0.0f},
                new float[]{100, 100, 50, 50, 0.08f, 0.1f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8.postProcess(output, shape(2),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertTrue(results.isEmpty());
    }

    @Test
    void postProcess_correctLabelAndClassIndex() {
        float[] output = createOutput(
                // Best class is "motorcycle" (index 3)
                new float[]{320, 320, 100, 100, 0.1f, 0.2f, 0.3f, 0.8f, 0.1f}
        );

        List<Detection> results = YoloV8.postProcess(output, shape(1),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        assertEquals("motorcycle", results.get(0).label());
        assertEquals(3, results.get(0).classIndex());
        assertEquals(0.8f, results.get(0).confidence(), 1e-5f);
    }

    @Test
    void postProcess_clipsCoordinatesToImageBounds() {
        // Box extends beyond image boundaries
        float[] output = createOutput(
                new float[]{10, 10, 100, 100, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8.postProcess(output, shape(1),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        BoundingBox box = results.get(0).box();
        assertTrue(box.x1() >= 0, "x1 should be >= 0, was " + box.x1());
        assertTrue(box.y1() >= 0, "y1 should be >= 0, was " + box.y1());
        assertTrue(box.x2() <= ORIG_W, "x2 should be <= " + ORIG_W + ", was " + box.x2());
        assertTrue(box.y2() <= ORIG_H, "y2 should be <= " + ORIG_H + ", was " + box.y2());
    }
}
