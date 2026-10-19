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
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import io.github.inference4j.InferenceSession;
import io.github.inference4j.Tensor;
import io.github.inference4j.exception.ModelSourceException;
import io.github.inference4j.model.ModelSource;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class Sam2SegmenterTest {

    // --- Preprocessing ---

    @Test
    void imageToTensorStretchesToModelInput() {
        BufferedImage image = new BufferedImage(64, 32, BufferedImage.TYPE_INT_RGB);

        Tensor tensor = Sam2Segmenter.imageToTensor(image);

        assertThat(tensor.shape()).containsExactly(1, 3, 1024, 1024);
    }

    @Test
    void imageToTensorAppliesImageNetNormalization() {
        BufferedImage image = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                image.setRGB(x, y, 0xFFFFFF);
            }
        }

        float[] data = Sam2Segmenter.imageToTensor(image).toFloats();

        int plane = 1024 * 1024;
        assertThat(data[0]).isCloseTo((1f - 0.485f) / 0.229f, within(1e-5f));
        assertThat(data[plane]).isCloseTo((1f - 0.456f) / 0.224f, within(1e-5f));
        assertThat(data[2 * plane]).isCloseTo((1f - 0.406f) / 0.225f, within(1e-5f));
    }

    @Test
    void promptTensorsScalePointsIntoModelSpace() {
        Prompt prompt = Prompt.point(100, 50).withBackground(200, 25);

        Map<String, Tensor> tensors = Sam2Segmenter.promptTensors(prompt, 400, 200);

        assertThat(tensors.get("input_points").shape()).containsExactly(1, 1, 2, 2);
        assertThat(tensors.get("input_points").toFloats()).containsExactly(256f, 256f, 512f, 128f);
        assertThat(tensors.get("input_labels").toLongs()).containsExactly(1L, 0L);
        assertThat(tensors.get("input_boxes").shape()).containsExactly(1, 0, 4);
    }

    @Test
    void promptTensorsScaleBoxIntoModelSpace() {
        Prompt prompt = Prompt.box(40, 20, 200, 100);

        Map<String, Tensor> tensors = Sam2Segmenter.promptTensors(prompt, 400, 200);

        assertThat(tensors.get("input_boxes").shape()).containsExactly(1, 1, 4);
        assertThat(tensors.get("input_boxes").toFloats())
                .containsExactly(new float[]{102.4f, 102.4f, 512f, 512f}, within(1e-4f));
        assertThat(tensors.get("input_points").shape()).containsExactly(1, 1, 0, 2);
        assertThat(tensors.get("input_labels").shape()).containsExactly(1, 1, 0);
    }

    // --- Post-processing ---

    @Test
    void resizeBilinearMatchesPyTorchAlignCornersFalse() {
        // torch.nn.functional.interpolate(t, size=(4, 4), mode="bilinear", align_corners=False)
        float[] result = Sam2Segmenter.resizeBilinear(new float[]{0f, 1f, 2f, 3f}, 2, 2, 4, 4);

        assertThat(result).containsExactly(new float[]{
                0.0f, 0.25f, 0.75f, 1.0f,
                0.5f, 0.75f, 1.25f, 1.5f,
                1.5f, 1.75f, 2.25f, 2.5f,
                2.0f, 2.25f, 2.75f, 3.0f}, within(1e-6f));
    }

    @Test
    void resizeBilinearDownsamplesWithoutAntialiasing() {
        // torch: interpolate(arange(16).view(1, 1, 4, 4), size=(2, 2), mode="bilinear", align_corners=False)
        float[] src = new float[16];
        for (int i = 0; i < 16; i++) {
            src[i] = i;
        }

        float[] result = Sam2Segmenter.resizeBilinear(src, 4, 4, 2, 2);

        assertThat(result).containsExactly(new float[]{2.5f, 4.5f, 10.5f, 12.5f}, within(1e-6f));
    }

    @Test
    void toMaskThresholdsLogitsAtZeroInOriginalSize() {
        float[] logits = {-5f, -5f, 5f, 5f};

        SegmentationMask mask = Sam2Segmenter.toMask(logits, 2, 2, 30, 20, 0.8f);

        assertThat(mask.width()).isEqualTo(30);
        assertThat(mask.height()).isEqualTo(20);
        assertThat(mask.score()).isEqualTo(0.8f);
        assertThat(mask.contains(15, 2)).isFalse();
        assertThat(mask.contains(15, 17)).isTrue();
    }

    // --- End-to-End ---

    @Test
    void segmentReturnsHighestScoringMaskAtImageSize() {
        try (Sam2Segmenter sam = segmenterReturning(new float[]{0.2f, 0.9f, 0.5f})) {
            BufferedImage image = new BufferedImage(40, 30, BufferedImage.TYPE_INT_RGB);

            SegmentationMask mask = sam.segment(sam.encode(image), Prompt.point(10, 10));

            assertThat(mask.score()).isEqualTo(0.9f);
            assertThat(mask.width()).isEqualTo(40);
            assertThat(mask.height()).isEqualTo(30);
            assertThat(mask.area()).isEqualTo(40 * 30);
        }
    }

    @Test
    void segmentCandidatesAreSortedByScore() {
        try (Sam2Segmenter sam = segmenterReturning(new float[]{0.2f, 0.9f, 0.5f})) {
            ImageFeatures features = sam.encode(new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB));

            List<SegmentationMask> masks = sam.segmentCandidates(features, Prompt.point(1, 1));

            assertThat(masks).extracting(SegmentationMask::score).containsExactly(0.9f, 0.5f, 0.2f);
        }
    }

    @Test
    void segmentPassesEmbeddingsAndPromptToDecoder() {
        InferenceSession encoder = mockEncoder();
        InferenceSession decoder = mockDecoder(new float[]{0.1f, 0.2f, 0.3f});
        try (Sam2Segmenter sam = Sam2Segmenter.builder()
                .encoderSession(encoder)
                .decoderSession(decoder)
                .build()) {
            ImageFeatures features = sam.encode(new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB));
            sam.segment(features, Prompt.box(0, 0, 4, 4));
        }

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, Tensor>> captor = ArgumentCaptor.forClass(Map.class);
        verify(decoder).run(captor.capture());
        assertThat(captor.getValue()).containsKeys(
                "image_embeddings.0", "image_embeddings.1", "image_embeddings.2",
                "input_points", "input_labels", "input_boxes");
        assertThat(captor.getValue().get("input_boxes").toFloats()).containsExactly(0f, 0f, 512f, 512f);
    }

    @Test
    void encodeKeepsImageSize() {
        try (Sam2Segmenter sam = segmenterReturning(new float[]{0.1f, 0.2f, 0.3f})) {
            ImageFeatures features = sam.encode(new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB));

            assertThat(features.width()).isEqualTo(640);
            assertThat(features.height()).isEqualTo(480);
        }
    }

    @Test
    void closeClosesBothSessions() {
        InferenceSession encoder = mockEncoder();
        InferenceSession decoder = mockDecoder(new float[]{0.1f, 0.2f, 0.3f});

        Sam2Segmenter.builder().encoderSession(encoder).decoderSession(decoder).build().close();

        verify(encoder).close();
        verify(decoder).close();
    }

    @Test
    void builderWithMissingModelDirectoryThrows() {
        ModelSource badSource = id -> Path.of("/nonexistent/path/" + id);

        assertThatThrownBy(() -> Sam2Segmenter.builder().modelSource(badSource).build())
                .isInstanceOf(ModelSourceException.class);
    }

    // --- Helpers ---

    /** A segmenter whose decoder proposes three 2x2 masks: only the second is all foreground. */
    private static Sam2Segmenter segmenterReturning(float[] scores) {
        return Sam2Segmenter.builder()
                .encoderSession(mockEncoder())
                .decoderSession(mockDecoder(scores))
                .build();
    }

    private static InferenceSession mockEncoder() {
        InferenceSession encoder = mock(InferenceSession.class);
        Tensor tiny = Tensor.fromFloats(new float[]{0f}, new long[]{1, 1, 1, 1});
        when(encoder.run(any())).thenReturn(Map.of(
                "image_embeddings.0", tiny,
                "image_embeddings.1", tiny,
                "image_embeddings.2", tiny));
        return encoder;
    }

    private static InferenceSession mockDecoder(float[] scores) {
        InferenceSession decoder = mock(InferenceSession.class);
        float[] masks = {
                -1f, -1f, -1f, -1f,
                1f, 1f, 1f, 1f,
                -1f, 1f, -1f, 1f};
        when(decoder.run(any())).thenReturn(Map.of(
                "iou_scores", Tensor.fromFloats(scores, new long[]{1, 1, 3}),
                "pred_masks", Tensor.fromFloats(masks, new long[]{1, 1, 3, 2, 2})));
        return decoder;
    }
}
