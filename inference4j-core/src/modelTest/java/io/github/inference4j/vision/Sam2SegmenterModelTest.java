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
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import static org.assertj.core.api.Assertions.*;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class Sam2SegmenterModelTest {

    private Sam2Segmenter sam;
    private BufferedImage catImage;
    private ImageFeatures catFeatures;

    @BeforeAll
    void setUp() throws IOException {
        sam = Sam2Segmenter.builder().build();
        catImage = ImageIO.read(
                Sam2SegmenterModelTest.class.getResourceAsStream("/fixtures/cat.jpg"));
        catFeatures = sam.encode(catImage);
    }

    @AfterAll
    void tearDown() {
        if (sam != null) sam.close();
    }

    @Test
    void pointPromptSegmentsTheClickedObject() {
        SegmentationMask mask = sam.segment(catFeatures, Prompt.point(160, 160));

        assertThat(mask.width()).isEqualTo(catImage.getWidth());
        assertThat(mask.height()).isEqualTo(catImage.getHeight());
        assertThat(mask.contains(160, 160)).isTrue();
        assertThat(mask.score()).isGreaterThan(0.9f);
        // Reference: ONNX Runtime + PyTorch post-processing gives 59,859 pixels for this prompt.
        assertThat(mask.area()).isBetween(55_000, 65_000);
    }

    @Test
    void boxPromptStaysInsideTheBox() {
        BoundingBox box = new BoundingBox(64, 32, 256, 288);

        SegmentationMask mask = sam.segment(catFeatures, Prompt.box(box));

        assertThat(mask.isEmpty()).isFalse();
        int inside = 0;
        for (int y = 0; y < mask.height(); y++) {
            for (int x = 0; x < mask.width(); x++) {
                if (mask.contains(x, y) && x >= box.x1() && x < box.x2() && y >= box.y1() && y < box.y2()) {
                    inside++;
                }
            }
        }
        // The cat fills the frame, so the box cuts through it; a few edge pixels may spill over.
        assertThat((double) inside / mask.area()).isGreaterThan(0.95);
    }

    @Test
    void candidatesAreThreeMasksBestFirst() {
        List<SegmentationMask> masks = sam.segmentCandidates(catFeatures, Prompt.point(160, 160));

        assertThat(masks).hasSize(3);
        assertThat(masks).extracting(SegmentationMask::score)
                .isSortedAccordingTo((a, b) -> Float.compare(b, a));
    }

    @Test
    void backgroundClickRemovesPixelsFromTheMask() {
        SegmentationMask base = sam.segment(catFeatures, Prompt.point(160, 160));

        SegmentationMask trimmed = sam.segment(catFeatures,
                Prompt.point(160, 160).withBackground(160, 300));

        assertThat(trimmed.contains(160, 160)).isTrue();
        assertThat(trimmed.area()).isLessThanOrEqualTo(base.area());
    }

    @Test
    void cutoutIsTransparentOutsideTheMask() {
        SegmentationMask mask = sam.segment(catFeatures, Prompt.point(160, 160));

        BufferedImage cutout = mask.applyTo(catImage);

        assertThat(cutout.getRGB(160, 160) >>> 24).isEqualTo(0xFF);
        assertThat(cutout.getWidth()).isEqualTo(catImage.getWidth());
    }
}
