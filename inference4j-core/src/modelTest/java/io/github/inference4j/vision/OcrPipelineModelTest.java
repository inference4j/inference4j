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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Expected text is what the {@code paddleocr} Python package reads from the same images with the same models
 * (PP-OCRv6, document orientation, unwarping and text-line orientation off).
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class OcrPipelineModelTest {

    private static final List<String> DOCUMENT_LINES = List.of(
            "Quarterly Transport Report",
            "The council approved a new budget for buses and road repairs.",
            "Two new routes will serve the northern suburbs from March.",
            "Fares stay at 2.40 for adults and 1.20 for students.",
            "Contact: transport@millbrook.example, phone 0161 555 0199",
            "Repairs are scheduled for the high street",
            "Published 14 October 2026 - page 1 of 3");

    private OcrPipeline ocr;

    @BeforeAll
    void setUp() {
        ocr = OcrPipeline.builder().build();
    }

    @AfterAll
    void tearDown() {
        if (ocr != null) {
            ocr.close();
        }
    }

    @Test
    void readsADocumentLineByLineInReadingOrder() throws Exception {
        List<TextLine> lines = ocr.read(resource("ocr-document.png"));

        assertThat(lines).extracting(TextLine::text).containsExactlyElementsOf(DOCUMENT_LINES);
        assertThat(lines).allSatisfy(line -> assertThat(line.confidence()).isGreaterThan(0.9f));
    }

    @Test
    void keepsTheOutlineOfARotatedLine() throws Exception {
        TextLine rotated = ocr.read(resource("ocr-document.png")).get(5);

        List<Point> corners = rotated.region().polygon();
        assertThat(corners).hasSize(4);
        // The line rises to the right: its top-right corner is well above its top-left one
        assertThat(corners.get(0).y() - corners.get(1).y()).isGreaterThan(80f);
    }

    @Test
    void readsLightTextOnColorAndATiltedLine() throws Exception {
        List<TextLine> lines = ocr.read(resource("ocr-signage.png"));

        assertThat(lines).extracting(TextLine::text)
                .containsExactly("PLATFORM 4", "Next train to Leeds 18:42", "Mind the gap");
    }

    @Test
    void tinyModelsReadTheDocument() throws Exception {
        try (var tiny = OcrPipeline.builder()
                .detector(PaddleTextDetector.tiny().build())
                .recognizer(PaddleTextRecognizer.tiny().build())
                .build()) {
            List<TextLine> lines = tiny.read(resource("ocr-document.png"));

            assertThat(lines).extracting(TextLine::text).containsExactlyElementsOf(DOCUMENT_LINES);
        }
    }

    @Test
    void detectorAndRecognizerWorkOnTheirOwn() throws Exception {
        BufferedImage page = ImageIO.read(resource("ocr-document.png").toFile());
        try (var detector = PaddleTextDetector.small().build();
             var recognizer = PaddleTextRecognizer.small().build()) {
            List<TextRegion> regions = detector.detect(page);
            RecognizedText first = recognizer.recognize(QuadCrop.crop(page, regions.get(0).polygon()));

            assertThat(regions).hasSize(7);
            assertThat(first.text()).isEqualTo("Quarterly Transport Report");
        }
    }

    @Test
    void craftCanReplaceTheDetector() throws Exception {
        try (var withCraft = OcrPipeline.builder()
                .detector(CraftTextDetector.builder().build())
                .recognizer(PaddleTextRecognizer.small().build())
                .build()) {
            List<TextLine> words = withCraft.read(resource("ocr-document.png"));

            // CRAFT finds words or word groups rather than lines
            String text = String.join(" ", words.stream().map(TextLine::text).toList());
            assertThat(text).contains("Quarterly", "Transport", "Report", "council");
        }
    }

    private Path resource(String name) throws URISyntaxException, IOException {
        return Path.of(Objects.requireNonNull(getClass().getResource("/" + name)).toURI());
    }
}
