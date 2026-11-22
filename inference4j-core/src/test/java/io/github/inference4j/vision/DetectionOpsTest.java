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

import io.github.inference4j.preprocessing.image.Labels;
import io.github.inference4j.vision.DetectionOps.Letterbox;
import io.github.inference4j.vision.DetectionOps.QueryDecoding;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.tuple;
import static org.assertj.core.data.Offset.offset;

class DetectionOpsTest {

    private static final Labels LABELS = Labels.of(List.of("cat", "dog", "bird"));

    // --- Letterbox ---

    @Test
    void centeredLetterboxSplitsThePaddingEvenly() {
        Letterbox letterbox = Letterbox.centered(200, 100, 64);

        assertThat(letterbox.scale()).isEqualTo(0.32f);
        assertThat(letterbox.scaledWidth()).isEqualTo(64);
        assertThat(letterbox.scaledHeight()).isEqualTo(32);
        assertThat(letterbox.padX()).isZero();
        assertThat(letterbox.padY()).isEqualTo(16f);
    }

    @Test
    void topLeftLetterboxPadsOnlyRightAndBottom() {
        Letterbox letterbox = Letterbox.topLeft(100, 200, 64);

        assertThat(letterbox.scaledWidth()).isEqualTo(32);
        assertThat(letterbox.scaledHeight()).isEqualTo(64);
        assertThat(letterbox.padX()).isZero();
        assertThat(letterbox.padY()).isZero();
    }

    @Test
    void topLeftLetterboxTruncatesTheScaledSizeLikeYolox() {
        // 230 * 416 / 352 = 271.8: YOLOX's int() keeps 271
        Letterbox letterbox = Letterbox.topLeft(352, 230, 416);

        assertThat(letterbox.scaledWidth()).isEqualTo(416);
        assertThat(letterbox.scaledHeight()).isEqualTo(271);
    }

    @Test
    void applyDrawsTheImageOnThePaddedCanvas() {
        BufferedImage white = new BufferedImage(200, 100, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 100; y++) {
            for (int x = 0; x < 200; x++) {
                white.setRGB(x, y, 0xFFFFFF);
            }
        }

        BufferedImage result = Letterbox.centered(200, 100, 64).apply(white, new Color(114, 114, 114));

        assertThat(result.getWidth()).isEqualTo(64);
        assertThat(result.getHeight()).isEqualTo(64);
        assertThat(result.getRGB(32, 5) & 0xFFFFFF).isEqualTo(0x727272);    // padding above
        assertThat(result.getRGB(32, 32) & 0xFFFFFF).isEqualTo(0xFFFFFF);   // image
        assertThat(result.getRGB(32, 60) & 0xFFFFFF).isEqualTo(0x727272);   // padding below
    }

    @Test
    void toOriginalRemovesPaddingAndScaleAndClamps() {
        Letterbox letterbox = Letterbox.centered(200, 100, 64);

        BoundingBox box = letterbox.toOriginal(new float[]{16f, 24f, 32f, 40f}, 200, 100);
        BoundingBox outside = letterbox.toOriginal(new float[]{-10f, 0f, 70f, 64f}, 200, 100);

        assertBox(box, 50f, 25f, 100f, 75f);
        assertThat(outside).isEqualTo(new BoundingBox(0f, 0f, 200f, 100f));
    }

    // --- Image to tensor ---

    @Test
    void toChwSplitsChannelsIntoPlanesScaledToUnitRange() {
        int[] pixels = {0xFF0000, 0x00FF00, 0x0000FF, 0x336699};

        float[] chw = DetectionOps.toChw(pixels, 2, 2);

        assertThat(chw).containsExactly(
                1f, 0f, 0f, 0x33 / 255f,
                0f, 1f, 0f, 0x66 / 255f,
                0f, 0f, 1f, 0x99 / 255f);
    }

    @Test
    void toChwNormalizesAndReordersChannels() {
        int[] pixels = {0xFF0000};

        float[] chw = DetectionOps.toChw(pixels, 1, 1, new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f},
                true);

        assertThat(chw).containsExactly(-1f, -1f, 1f);   // blue, green, red
    }

    @Test
    void toChwRejectsMeanOrStdWithoutThreeValues() {
        assertThatThrownBy(() -> DetectionOps.toChw(new int[1], 1, 1, new float[]{0f}, new float[]{1f, 1f, 1f}, false))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void pixelsReturnsPackedRgbRowByRow() {
        BufferedImage image = new BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB);
        image.setRGB(0, 0, 0x112233);
        image.setRGB(1, 0, 0x445566);

        int[] pixels = DetectionOps.pixels(image);

        assertThat(new int[]{pixels[0] & 0xFFFFFF, pixels[1] & 0xFFFFFF}).containsExactly(0x112233, 0x445566);
    }

    // --- NMS ---

    @Test
    void classAgnosticNmsSuppressesOverlapsAcrossClasses() {
        List<Detection> candidates = List.of(
                detection(0, 0.6f, 0, 0, 10, 10),
                detection(1, 0.9f, 1, 1, 11, 11),
                detection(0, 0.8f, 50, 50, 60, 60));

        List<Detection> kept = DetectionOps.nms(candidates, 0.5f, false, Integer.MAX_VALUE);

        assertThat(kept).extracting(Detection::confidence).containsExactly(0.9f, 0.8f);
    }

    @Test
    void classAwareNmsKeepsOverlappingBoxesOfDifferentClasses() {
        List<Detection> candidates = List.of(
                detection(0, 0.6f, 0, 0, 10, 10),
                detection(1, 0.9f, 1, 1, 11, 11),
                detection(1, 0.7f, 0, 0, 10, 10));

        List<Detection> kept = DetectionOps.nms(candidates, 0.5f, true, Integer.MAX_VALUE);

        assertThat(kept).extracting(Detection::label, Detection::confidence)
                .containsExactly(tuple("dog", 0.9f), tuple("cat", 0.6f));
    }

    @Test
    void nmsReturnsAtMostMaxDetections() {
        List<Detection> candidates = List.of(
                detection(0, 0.5f, 0, 0, 10, 10),
                detection(0, 0.7f, 20, 20, 30, 30),
                detection(0, 0.9f, 40, 40, 50, 50));

        assertThat(DetectionOps.nms(candidates, 0.5f, false, 2)).extracting(Detection::confidence)
                .containsExactly(0.9f, 0.7f);
        assertThat(DetectionOps.nms(List.of(), 0.5f, false, 2)).isEmpty();
    }

    // --- Query decoding ---

    private static final float HIGH = 3f;      // sigmoid ≈ 0.953
    private static final float MEDIUM = 1f;    // sigmoid ≈ 0.731
    private static final float LOW = -3f;      // sigmoid ≈ 0.047

    @Test
    void topPairsCanYieldSeveralClassesFromOneQuery() {
        float[] logits = {HIGH, MEDIUM, LOW, LOW, LOW, LOW};
        float[] boxes = {0.5f, 0.5f, 0.2f, 0.4f, 0.1f, 0.1f, 0.1f, 0.1f};

        List<Detection> detections = DetectionOps.decodeQueries(logits, boxes, 2, 3, 0.5f, 100, 50, LABELS,
                QueryDecoding.TOP_PAIRS, Integer.MAX_VALUE);

        assertThat(detections).extracting(Detection::label).containsExactly("cat", "dog");
        assertBox(detections.get(0).box(), 40f, 15f, 60f, 35f);
    }

    @Test
    void bestClassPerQueryKeepsOneClassPerQuery() {
        float[] logits = {HIGH, MEDIUM, LOW, LOW, LOW, MEDIUM};
        float[] boxes = {0.5f, 0.5f, 0.2f, 0.4f, 0.5f, 0.5f, 1f, 1f};

        List<Detection> detections = DetectionOps.decodeQueries(logits, boxes, 2, 3, 0.5f, 100, 50, LABELS,
                QueryDecoding.BEST_CLASS_PER_QUERY, Integer.MAX_VALUE);

        assertThat(detections).extracting(Detection::label).containsExactly("cat", "bird");
        assertThat(detections.get(1).box()).isEqualTo(new BoundingBox(0f, 0f, 100f, 50f));
    }

    @Test
    void decodingSortsByConfidenceAndCapsTheCount() {
        float[] logits = {MEDIUM, LOW, LOW, HIGH, LOW, LOW};
        float[] boxes = new float[8];

        List<Detection> detections = DetectionOps.decodeQueries(logits, boxes, 2, 3, 0.5f, 10, 10, LABELS,
                QueryDecoding.TOP_PAIRS, 1);

        assertThat(detections).singleElement()
                .satisfies(d -> assertThat(d.confidence()).isGreaterThan(0.9f));
    }

    @Test
    void thresholdIsExclusiveForTopPairsAndInclusiveForBestClass() {
        float[] logits = {0f};   // sigmoid = 0.5 exactly
        float[] boxes = new float[4];

        assertThat(DetectionOps.decodeQueries(logits, boxes, 1, 1, 0.5f, 10, 10, LABELS,
                QueryDecoding.TOP_PAIRS, Integer.MAX_VALUE)).isEmpty();
        assertThat(DetectionOps.decodeQueries(logits, boxes, 1, 1, 0.5f, 10, 10, LABELS,
                QueryDecoding.BEST_CLASS_PER_QUERY, Integer.MAX_VALUE)).hasSize(1);
    }

    private static void assertBox(BoundingBox box, float x1, float y1, float x2, float y2) {
        assertThat(box.x1()).isCloseTo(x1, offset(1e-4f));
        assertThat(box.y1()).isCloseTo(y1, offset(1e-4f));
        assertThat(box.x2()).isCloseTo(x2, offset(1e-4f));
        assertThat(box.y2()).isCloseTo(y2, offset(1e-4f));
    }

    private static Detection detection(int classIndex, float confidence, float x1, float y1, float x2, float y2) {
        return new Detection(new BoundingBox(x1, y1, x2, y2), LABELS.get(classIndex), classIndex, confidence);
    }
}
