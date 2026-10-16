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

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OcrPipelineTest {

    private final BufferedImage page = new BufferedImage(200, 100, BufferedImage.TYPE_INT_RGB);
    private final TextDetector detector = mock(TextDetector.class);
    private final TextRecognizer recognizer = mock(TextRecognizer.class);

    @Test
    void readsEveryRegionInReadingOrder() {
        TextRegion right = region(120, 10, 180, 30);
        TextRegion left = region(10, 12, 90, 30);      // same line: tops 2 px apart
        TextRegion below = region(10, 50, 150, 70);
        when(detector.detect(page)).thenReturn(List.of(below, right, left));
        when(recognizer.recognize(any(BufferedImage.class))).thenAnswer(call -> {
            BufferedImage crop = call.getArgument(0);
            return new RecognizedText("w" + crop.getWidth(), 0.9f);
        });

        List<TextLine> lines = pipeline(0f).read(page);

        assertThat(lines).extracting(TextLine::region).containsExactly(left, right, below);
        assertThat(lines).extracting(TextLine::text).containsExactly("w80", "w60", "w140");
    }

    @Test
    void leavesOutEmptyAndLowConfidenceLines() {
        TextRegion empty = region(10, 10, 50, 30);
        TextRegion unsure = region(10, 40, 50, 60);
        TextRegion sure = region(10, 70, 50, 90);
        when(detector.detect(page)).thenReturn(List.of(empty, unsure, sure));
        when(recognizer.recognize(any(BufferedImage.class)))
                .thenReturn(new RecognizedText("", 0f), new RecognizedText("maybe", 0.4f), new RecognizedText("yes", 0.95f));

        List<TextLine> lines = pipeline(0.5f).read(page);

        assertThat(lines).singleElement().satisfies(line -> {
            assertThat(line.text()).isEqualTo("yes");
            assertThat(line.confidence()).isEqualTo(0.95f);
            assertThat(line.region()).isEqualTo(sure);
        });
    }

    @Test
    void passesEachRegionCroppedToTheRecognizer() {
        when(detector.detect(page)).thenReturn(List.of(region(20, 30, 120, 50)));
        when(recognizer.recognize(any(BufferedImage.class))).thenReturn(new RecognizedText("x", 1f));

        pipeline(0f).read(page);

        verify(recognizer).recognize(argThat((BufferedImage crop) -> crop.getWidth() == 100 && crop.getHeight() == 20));
    }

    @Test
    void closingThePipelineClosesBothStages() {
        pipeline(0f).close();

        verify(detector).close();
        verify(recognizer).close();
    }

    private OcrPipeline pipeline(float minConfidence) {
        return OcrPipeline.builder().detector(detector).recognizer(recognizer).minConfidence(minConfidence).build();
    }

    private static TextRegion region(float x1, float y1, float x2, float y2) {
        return new TextRegion(new BoundingBox(x1, y1, x2, y2), 0.9f);
    }
}
