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
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.Metadata;
import io.github.inference4j.exception.InferenceException;
import io.github.inference4j.vision.OcrPipeline;
import io.github.inference4j.vision.TextLine;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * A LangChain4j {@link DocumentParser} that reads the text in an image with inference4j OCR, in-process, with no
 * OCR server to run:
 *
 * <pre>{@code
 * try (var parser = new Inference4jOcrDocumentParser(OcrPipeline.builder().build())) {
 *     Document invoice = FileSystemDocumentLoader.loadDocument(Path.of("invoice.png"), parser);
 * }
 * }</pre>
 *
 * <p>The document holds one line of text per detected line, in reading order (top to bottom, left to right), and
 * carries {@value #LINE_COUNT} and {@value #MEAN_CONFIDENCE} metadata. Lines are ordered by position only, so
 * multi-column pages and tables come out row by row rather than column by column.
 *
 * <p>Input is any image {@link ImageIO} can read; the JDK reads PNG, JPEG, BMP and GIF. Scanned PDFs need
 * rasterizing into images first, for example with PDFBox.
 *
 * <p>The parser owns the pipeline: {@link #close()} closes it. Thread-safe: one parser can serve a whole
 * application.
 */
public class Inference4jOcrDocumentParser implements DocumentParser, AutoCloseable {

    /** Number of text lines read from the image ({@code int}). */
    public static final String LINE_COUNT = "ocr_line_count";

    /** Mean recognition confidence of those lines, between 0 and 1 ({@code double}). */
    public static final String MEAN_CONFIDENCE = "ocr_mean_confidence";

    private final OcrPipeline ocr;

    /**
     * @param ocr the OCR pipeline to read images with; closed by {@link #close()}
     */
    public Inference4jOcrDocumentParser(OcrPipeline ocr) {
        this.ocr = Objects.requireNonNull(ocr, "ocr is required");
    }

    /**
     * Reads the text in the image into one {@link Document}, one line per detected text line.
     *
     * @throws InferenceException if the stream is not a readable image, or no text is found (LangChain4j documents
     *                            cannot be blank)
     */
    @Override
    public Document parse(InputStream inputStream) {
        List<TextLine> lines = ocr.read(readImage(inputStream));
        String text = lines.stream().map(TextLine::text).collect(Collectors.joining("\n"));
        if (text.isBlank()) {
            throw new InferenceException("No text found in image");
        }
        double meanConfidence = lines.stream().mapToDouble(TextLine::confidence).average().orElse(0);
        Metadata metadata = new Metadata()
                .put(LINE_COUNT, lines.size())
                .put(MEAN_CONFIDENCE, meanConfidence);
        return Document.from(text, metadata);
    }

    /**
     * Closes the OCR pipeline.
     */
    @Override
    public void close() {
        ocr.close();
    }

    private static BufferedImage readImage(InputStream inputStream) {
        try {
            BufferedImage image = ImageIO.read(inputStream);
            if (image == null) {
                throw new InferenceException("Unsupported image format");
            }
            return image;
        } catch (IOException e) {
            throw new InferenceException("Failed to read image: " + e.getMessage(), e);
        }
    }
}
