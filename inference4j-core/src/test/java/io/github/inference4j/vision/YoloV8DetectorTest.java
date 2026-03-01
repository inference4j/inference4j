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
import io.github.inference4j.preprocessing.image.Labels;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class YoloV8DetectorTest {

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

        List<Detection> results = YoloV8Detector.postProcess(output, shape(2),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertThat(results).hasSize(1);
        assertThat(results.get(0).label()).isEqualTo("person");
    }

    @Test
    void postProcess_correctBoxConversion() {
        // cx=320, cy=240, w=200, h=100 → x1=220, y1=190, x2=420, y2=290
        float[] output = createOutput(
                new float[]{320, 240, 200, 100, 0.8f, 0.0f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8Detector.postProcess(output, shape(1),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertThat(results).hasSize(1);
        BoundingBox box = results.get(0).box();
        assertThat(box.x1()).isCloseTo(220f, within(1e-3f));
        assertThat(box.y1()).isCloseTo(190f, within(1e-3f));
        assertThat(box.x2()).isCloseTo(420f, within(1e-3f));
        assertThat(box.y2()).isCloseTo(290f, within(1e-3f));
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

        List<Detection> results = YoloV8Detector.postProcess(output, shape(1),
                TEST_LABELS, 0.25f, 0.45f,
                scale, padX, padY, origW, origH);

        assertThat(results).hasSize(1);
        BoundingBox box = results.get(0).box();
        // cx=320, w=100 → x1=270, x2=370 in letterbox coords
        // rescaled: x1=(270-0)/0.5=540, x2=(370-0)/0.5=740
        assertThat(box.x1()).isCloseTo(540f, within(1e-1f));
        assertThat(box.x2()).isCloseTo(740f, within(1e-1f));
        // cy=320, h=60 → y1=290, y2=350 in letterbox coords
        // rescaled: y1=(290-140)/0.5=300, y2=(350-140)/0.5=420
        assertThat(box.y1()).isCloseTo(300f, within(1e-1f));
        assertThat(box.y2()).isCloseTo(420f, within(1e-1f));
        assertThat(results.get(0).label()).isEqualTo("car");
    }

    @Test
    void postProcess_nmsSuppressesOverlappingBoxes() {
        // Two highly overlapping detections of the same object
        float[] output = createOutput(
                new float[]{320, 320, 100, 100, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f},
                new float[]{325, 325, 100, 100, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8Detector.postProcess(output, shape(2),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertThat(results).hasSize(1);
        assertThat(results.get(0).confidence()).isCloseTo(0.9f, within(1e-5f));
    }

    @Test
    void postProcess_sortedByConfidenceDescending() {
        // Three non-overlapping detections with different confidences
        float[] output = createOutput(
                new float[]{100, 100, 50, 50, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f},
                new float[]{300, 300, 50, 50, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f},
                new float[]{500, 500, 50, 50, 0.7f, 0.0f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8Detector.postProcess(output, shape(3),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertThat(results).hasSize(3);
        assertThat(results.get(0).confidence()).isCloseTo(0.9f, within(1e-5f));
        assertThat(results.get(1).confidence()).isCloseTo(0.7f, within(1e-5f));
        assertThat(results.get(2).confidence()).isCloseTo(0.5f, within(1e-5f));
    }

    @Test
    void postProcess_emptyWhenAllBelowThreshold() {
        float[] output = createOutput(
                new float[]{320, 320, 100, 100, 0.1f, 0.05f, 0.02f, 0.0f, 0.0f},
                new float[]{100, 100, 50, 50, 0.08f, 0.1f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8Detector.postProcess(output, shape(2),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertThat(results).isEmpty();
    }

    @Test
    void postProcess_correctLabelAndClassIndex() {
        float[] output = createOutput(
                // Best class is "motorcycle" (index 3)
                new float[]{320, 320, 100, 100, 0.1f, 0.2f, 0.3f, 0.8f, 0.1f}
        );

        List<Detection> results = YoloV8Detector.postProcess(output, shape(1),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertThat(results).hasSize(1);
        assertThat(results.get(0).label()).isEqualTo("motorcycle");
        assertThat(results.get(0).classIndex()).isEqualTo(3);
        assertThat(results.get(0).confidence()).isCloseTo(0.8f, within(1e-5f));
    }

    @Test
    void postProcess_clipsCoordinatesToImageBounds() {
        // Box extends beyond image boundaries
        float[] output = createOutput(
                new float[]{10, 10, 100, 100, 0.9f, 0.0f, 0.0f, 0.0f, 0.0f}
        );

        List<Detection> results = YoloV8Detector.postProcess(output, shape(1),
                TEST_LABELS, 0.25f, 0.45f,
                SCALE, PAD_X, PAD_Y, ORIG_W, ORIG_H);

        assertThat(results).hasSize(1);
        BoundingBox box = results.get(0).box();
        assertThat(box.x1()).as("x1 should be >= 0, was " + box.x1()).isGreaterThanOrEqualTo(0);
        assertThat(box.y1()).as("y1 should be >= 0, was " + box.y1()).isGreaterThanOrEqualTo(0);
        assertThat(box.x2()).as("x2 should be <= " + ORIG_W + ", was " + box.x2()).isLessThanOrEqualTo(ORIG_W);
        assertThat(box.y2()).as("y2 should be <= " + ORIG_H + ", was " + box.y2()).isLessThanOrEqualTo(ORIG_H);
    }

    // --- Builder validation ---

    @Test
    void builder_invalidModelSource_throws() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);
        assertThatThrownBy(() ->
                YoloV8Detector.builder()
                        .inputName("images")
                        .modelSource(badSource)
                        .build())
                .isInstanceOf(ModelSourceException.class);
    }

    @Test
    void builder_inputNameDefaultsFromSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("images"));

        YoloV8Detector model = YoloV8Detector.builder()
                .session(session)
                .build();

        assertThat(model).isNotNull();
        verify(session).inputNames();
    }

    // --- Inference flow ---

    @Test
    void detect_bufferedImage_returnsCorrectResults() {
        InferenceSession session = mock(InferenceSession.class);

        // Single candidate: person at center of 32x32 input space
        float[] output = createOutput(
                new float[]{16, 16, 20, 20, 0.9f, 0.1f, 0.0f, 0.0f, 0.0f}
        );
        when(session.run(any())).thenReturn(
                Map.of("output", Tensor.fromFloats(output, shape(1))));

        YoloV8Detector model = YoloV8Detector.builder()
                .session(session)
                .labels(TEST_LABELS)
                .inputName("images")
                .inputSize(32)
                .build();

        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        List<Detection> results = model.detect(image);

        assertThat(results).hasSize(1);
        assertThat(results.get(0).label()).isEqualTo("person");
        assertThat(results.get(0).confidence()).isCloseTo(0.9f, within(1e-5f));
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);

        YoloV8Detector model = YoloV8Detector.builder()
                .session(session)
                .inputName("images")
                .build();

        model.close();

        verify(session).close();
    }
}
