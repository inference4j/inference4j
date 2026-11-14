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

package io.github.inference4j.langchain4j;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.vision.BoundingBox;
import io.github.inference4j.vision.OcrPipeline;
import io.github.inference4j.vision.Point;
import io.github.inference4j.vision.RecognizedText;
import io.github.inference4j.vision.TextDetector;
import io.github.inference4j.vision.TextRecognizer;
import io.github.inference4j.vision.TextRegion;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.github.inference4j.langchain4j.Inference4jOcrDocumentParser.LINE_COUNT;
import static io.github.inference4j.langchain4j.Inference4jOcrDocumentParser.MEAN_CONFIDENCE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

class Inference4jOcrDocumentParserTest {

    /** Returns fixed regions, in the order given. */
    private static final class FixedDetector implements TextDetector {

        private final List<TextRegion> regions;
        boolean closed;

        FixedDetector(TextRegion... regions) {
            this.regions = List.of(regions);
        }

        @Override
        public List<TextRegion> detect(BufferedImage image) {
            return regions;
        }

        @Override
        public List<TextRegion> detect(Path imagePath) {
            return regions;
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    /** Reads each crop as the text registered for its width, so tests can tell lines apart. */
    private static final class WidthRecognizer implements TextRecognizer {

        private final Map<Integer, RecognizedText> byWidth;
        boolean closed;

        WidthRecognizer(Map<Integer, RecognizedText> byWidth) {
            this.byWidth = byWidth;
        }

        @Override
        public RecognizedText recognize(BufferedImage lineImage) {
            return byWidth.getOrDefault(lineImage.getWidth(), new RecognizedText("", 0f));
        }

        @Override
        public void close() {
            closed = true;
        }
    }

    /** An axis-aligned line region. */
    private static TextRegion line(int x, int y, int width, int height) {
        return new TextRegion(new BoundingBox(x, y, x + width, y + height),
                List.of(new Point(x, y), new Point(x + width, y), new Point(x + width, y + height),
                        new Point(x, y + height)),
                0.9f);
    }

    private static InputStream png() throws IOException {
        var out = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(200, 100, BufferedImage.TYPE_INT_RGB), "png", out);
        return new ByteArrayInputStream(out.toByteArray());
    }

    private static Inference4jOcrDocumentParser parser(TextDetector detector, TextRecognizer recognizer) {
        return new Inference4jOcrDocumentParser(OcrPipeline.builder().detector(detector).recognizer(recognizer).build());
    }

    private static final Map<Integer, RecognizedText> INVOICE = Map.of(
            120, new RecognizedText("INVOICE 2026-0042", 0.98f),
            80, new RecognizedText("Total: 41.50", 0.90f));

    // --- parse() ---

    @Test
    void linesAreJoinedInReadingOrder() throws IOException {
        var detector = new FixedDetector(line(10, 60, 80, 20), line(10, 10, 120, 20));   // bottom line first

        Document document = parser(detector, new WidthRecognizer(INVOICE)).parse(png());

        assertThat(document.text()).isEqualTo("INVOICE 2026-0042\nTotal: 41.50");
    }

    @Test
    void metadataGivesLineCountAndMeanConfidence() throws IOException {
        var detector = new FixedDetector(line(10, 10, 120, 20), line(10, 60, 80, 20));

        Document document = parser(detector, new WidthRecognizer(INVOICE)).parse(png());

        assertThat(document.metadata().getInteger(LINE_COUNT)).isEqualTo(2);
        assertThat(document.metadata().getDouble(MEAN_CONFIDENCE)).isCloseTo(0.94, within(1e-6));
    }

    @Test
    void linesReadAsEmptyAreLeftOut() throws IOException {
        var detector = new FixedDetector(line(10, 10, 120, 20), line(10, 60, 50, 20));   // width 50 reads as ""

        Document document = parser(detector, new WidthRecognizer(INVOICE)).parse(png());

        assertThat(document.text()).isEqualTo("INVOICE 2026-0042");
        assertThat(document.metadata().getInteger(LINE_COUNT)).isEqualTo(1);
    }

    @Test
    void anImageWithoutTextIsRejected() {
        var parser = parser(new FixedDetector(), new WidthRecognizer(INVOICE));

        assertThatThrownBy(() -> parser.parse(png()))
                .isInstanceOf(InferenceException.class)
                .hasMessage("No text found in image");
    }

    @Test
    void aStreamThatIsNotAnImageIsRejected() {
        var parser = parser(new FixedDetector(line(10, 10, 120, 20)), new WidthRecognizer(INVOICE));
        InputStream notAnImage = new ByteArrayInputStream("not an image".getBytes(StandardCharsets.UTF_8));

        assertThatThrownBy(() -> parser.parse(notAnImage))
                .isInstanceOf(InferenceException.class)
                .hasMessage("Unsupported image format");
    }

    @Test
    void worksWithFileSystemDocumentLoader(@TempDir Path dir) throws IOException {
        Path image = dir.resolve("invoice.png");
        try (InputStream in = png()) {
            Files.copy(in, image);
        }
        var parser = parser(new FixedDetector(line(10, 10, 120, 20)), new WidthRecognizer(INVOICE));

        Document document = FileSystemDocumentLoader.loadDocument(image, parser);

        assertThat(document.text()).isEqualTo("INVOICE 2026-0042");
        assertThat(document.metadata().getString(Document.FILE_NAME)).isEqualTo("invoice.png");
    }

    // --- lifecycle ---

    @Test
    void aPipelineIsRequired() {
        assertThatThrownBy(() -> new Inference4jOcrDocumentParser(null))
                .isInstanceOf(NullPointerException.class)
                .hasMessageContaining("ocr");
    }

    @Test
    void closeClosesBothPipelineStages() {
        var detector = new FixedDetector();
        var recognizer = new WidthRecognizer(INVOICE);

        parser(detector, recognizer).close();

        assertThat(detector.closed).isTrue();
        assertThat(recognizer.closed).isTrue();
    }
}
