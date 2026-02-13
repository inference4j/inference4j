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

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.image.Labels;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class Yolo26Test {

    // YOLO26 output: logits [1, numProposals, numClasses], boxes [1, numProposals, 4]
    private static final int NUM_CLASSES = 5;
    private static final int NUM_PROPOSALS = 300;

    private static final Labels TEST_LABELS = Labels.of(List.of(
            "person", "bicycle", "car", "motorcycle", "airplane"
    ));

    // Default image dimensions
    private static final int ORIG_W = 1280;
    private static final int ORIG_H = 720;

    /**
     * Creates synthetic logits array [1, numProposals, numClasses].
     * Each entry in proposals is {class0_logit, class1_logit, ...}.
     * Remaining proposals are filled with large negative values (low confidence after sigmoid).
     */
    private static float[] createLogits(float[]... proposals) {
        float[] logits = new float[NUM_PROPOSALS * NUM_CLASSES];
        // Fill with large negative values so sigmoid → ~0
        java.util.Arrays.fill(logits, -10f);
        for (int p = 0; p < proposals.length; p++) {
            for (int cls = 0; cls < NUM_CLASSES; cls++) {
                logits[p * NUM_CLASSES + cls] = proposals[p][cls];
            }
        }
        return logits;
    }

    /**
     * Creates synthetic boxes array [1, numProposals, 4].
     * Each entry is {cx, cy, w, h} in normalized 0–1 coords.
     * Remaining proposals are filled with zeros.
     */
    private static float[] createBoxes(float[]... proposals) {
        float[] boxes = new float[NUM_PROPOSALS * 4];
        for (int p = 0; p < proposals.length; p++) {
            System.arraycopy(proposals[p], 0, boxes, p * 4, 4);
        }
        return boxes;
    }

    private static long[] logitsShape() {
        return new long[]{1, NUM_PROPOSALS, NUM_CLASSES};
    }

    private static long[] boxesShape() {
        return new long[]{1, NUM_PROPOSALS, 4};
    }

    @Test
    void postProcess_filtersLowConfidenceDetections() {
        float[] logits = createLogits(
                // logit=2 → sigmoid≈0.88 (high), logit=-2 → sigmoid≈0.12 (low)
                new float[]{2.0f, -5f, -5f, -5f, -5f},   // person, high confidence
                new float[]{-2.0f, -3f, -5f, -5f, -5f}    // all low after sigmoid
        );
        float[] boxes = createBoxes(
                new float[]{0.5f, 0.5f, 0.2f, 0.2f},
                new float[]{0.2f, 0.2f, 0.1f, 0.1f}
        );

        List<Detection> results = Yolo26.postProcess(logits, logitsShape(),
                boxes, boxesShape(), TEST_LABELS, 0.5f, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        assertEquals("person", results.get(0).label());
    }

    @Test
    void postProcess_appliesSigmoidToLogits() {
        // logit=0 → sigmoid=0.5 exactly
        float[] logits = createLogits(
                new float[]{0f, -10f, -10f, -10f, -10f}
        );
        float[] boxes = createBoxes(
                new float[]{0.5f, 0.5f, 0.1f, 0.1f}
        );

        // With threshold just below 0.5, should pass
        List<Detection> results = Yolo26.postProcess(logits, logitsShape(),
                boxes, boxesShape(), TEST_LABELS, 0.49f, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        assertEquals(0.5f, results.get(0).confidence(), 1e-5f);

        // With threshold just above 0.5, should be filtered
        List<Detection> filtered = Yolo26.postProcess(logits, logitsShape(),
                boxes, boxesShape(), TEST_LABELS, 0.51f, ORIG_W, ORIG_H);

        assertTrue(filtered.isEmpty());
    }

    @Test
    void postProcess_correctCoordinateConversion() {
        // Normalized box: cx=0.5, cy=0.5, w=0.25, h=0.5
        // → x1=(0.5-0.125)*1280=480, y1=(0.5-0.25)*720=180
        //   x2=(0.5+0.125)*1280=800, y2=(0.5+0.25)*720=540
        float[] logits = createLogits(
                new float[]{3.0f, -5f, -5f, -5f, -5f}
        );
        float[] boxes = createBoxes(
                new float[]{0.5f, 0.5f, 0.25f, 0.5f}
        );

        List<Detection> results = Yolo26.postProcess(logits, logitsShape(),
                boxes, boxesShape(), TEST_LABELS, 0.5f, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        BoundingBox box = results.get(0).box();
        assertEquals(480f, box.x1(), 1e-1f);
        assertEquals(180f, box.y1(), 1e-1f);
        assertEquals(800f, box.x2(), 1e-1f);
        assertEquals(540f, box.y2(), 1e-1f);
    }

    @Test
    void postProcess_sortedByConfidenceDescending() {
        // Three proposals with different confidence levels
        // logit=1 → sigmoid≈0.73, logit=3 → sigmoid≈0.95, logit=2 → sigmoid≈0.88
        float[] logits = createLogits(
                new float[]{1.0f, -5f, -5f, -5f, -5f},
                new float[]{3.0f, -5f, -5f, -5f, -5f},
                new float[]{2.0f, -5f, -5f, -5f, -5f}
        );
        float[] boxes = createBoxes(
                new float[]{0.2f, 0.2f, 0.1f, 0.1f},
                new float[]{0.5f, 0.5f, 0.1f, 0.1f},
                new float[]{0.8f, 0.8f, 0.1f, 0.1f}
        );

        List<Detection> results = Yolo26.postProcess(logits, logitsShape(),
                boxes, boxesShape(), TEST_LABELS, 0.5f, ORIG_W, ORIG_H);

        assertEquals(3, results.size());
        assertTrue(results.get(0).confidence() > results.get(1).confidence());
        assertTrue(results.get(1).confidence() > results.get(2).confidence());
    }

    @Test
    void postProcess_emptyWhenAllBelowThreshold() {
        float[] logits = createLogits(
                new float[]{-3f, -4f, -5f, -5f, -5f},
                new float[]{-2f, -3f, -5f, -5f, -5f}
        );
        float[] boxes = createBoxes(
                new float[]{0.5f, 0.5f, 0.2f, 0.2f},
                new float[]{0.2f, 0.2f, 0.1f, 0.1f}
        );

        List<Detection> results = Yolo26.postProcess(logits, logitsShape(),
                boxes, boxesShape(), TEST_LABELS, 0.5f, ORIG_W, ORIG_H);

        assertTrue(results.isEmpty());
    }

    @Test
    void postProcess_correctLabelAndClassIndex() {
        // Best class is "motorcycle" (index 3): logit=4 → sigmoid≈0.98
        float[] logits = createLogits(
                new float[]{-5f, -5f, -5f, 4.0f, -5f}
        );
        float[] boxes = createBoxes(
                new float[]{0.5f, 0.5f, 0.2f, 0.2f}
        );

        List<Detection> results = Yolo26.postProcess(logits, logitsShape(),
                boxes, boxesShape(), TEST_LABELS, 0.5f, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        assertEquals("motorcycle", results.get(0).label());
        assertEquals(3, results.get(0).classIndex());
    }

    @Test
    void postProcess_clipsCoordinatesToImageBounds() {
        // Box that extends beyond image boundaries
        // cx=0.05, w=0.2 → x1=-0.05 (clipped to 0), x2=0.15
        float[] logits = createLogits(
                new float[]{3.0f, -5f, -5f, -5f, -5f}
        );
        float[] boxes = createBoxes(
                new float[]{0.05f, 0.05f, 0.2f, 0.2f}
        );

        List<Detection> results = Yolo26.postProcess(logits, logitsShape(),
                boxes, boxesShape(), TEST_LABELS, 0.5f, ORIG_W, ORIG_H);

        assertEquals(1, results.size());
        BoundingBox box = results.get(0).box();
        assertTrue(box.x1() >= 0, "x1 should be >= 0, was " + box.x1());
        assertTrue(box.y1() >= 0, "y1 should be >= 0, was " + box.y1());
        assertTrue(box.x2() <= ORIG_W, "x2 should be <= " + ORIG_W + ", was " + box.x2());
        assertTrue(box.y2() <= ORIG_H, "y2 should be <= " + ORIG_H + ", was " + box.y2());
    }

    // --- Builder validation ---

    @Test
    void builder_missingSession_throws() {
        assertThrows(IllegalStateException.class, () ->
                Yolo26.builder()
                        .inputName("images")
                        .build());
    }

    @Test
    void builder_inputNameDefaultsFromSession() {
        InferenceSession session = mock(InferenceSession.class);
        when(session.inputNames()).thenReturn(Set.of("images"));

        Yolo26 model = Yolo26.builder()
                .session(session)
                .build();

        assertNotNull(model);
        verify(session).inputNames();
    }

    // --- Inference flow ---

    @Test
    void detect_bufferedImage_returnsCorrectResults() {
        InferenceSession session = mock(InferenceSession.class);

        // Single proposal: person with logit=3 (sigmoid≈0.95), centered box
        float[] logits = new float[]{3.0f, -5f, -5f, -5f, -5f};
        float[] boxes = new float[]{0.5f, 0.5f, 0.25f, 0.25f};

        Map<String, Tensor> outputs = new LinkedHashMap<>();
        outputs.put("logits", Tensor.fromFloats(logits, new long[]{1, 1, NUM_CLASSES}));
        outputs.put("pred_boxes", Tensor.fromFloats(boxes, new long[]{1, 1, 4}));
        when(session.run(any())).thenReturn(outputs);

        Yolo26 model = Yolo26.builder()
                .session(session)
                .labels(TEST_LABELS)
                .inputName("images")
                .inputSize(32)
                .confidenceThreshold(0.5f)
                .build();

        BufferedImage image = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        List<Detection> results = model.detect(image);

        assertEquals(1, results.size());
        assertEquals("person", results.get(0).label());
        assertTrue(results.get(0).confidence() > 0.9f);
    }

    // --- Close delegation ---

    @Test
    void close_delegatesToSession() {
        InferenceSession session = mock(InferenceSession.class);

        Yolo26 model = Yolo26.builder()
                .session(session)
                .inputName("images")
                .build();

        model.close();

        verify(session).close();
    }
}
