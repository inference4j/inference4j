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

import io.github.inference4j.InferenceTask;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Optical character recognition: finds the lines of text in an image and reads them.
 *
 * <p>Recognizers read one line at a time, so the pipeline runs a {@link TextDetector} first, cuts each detected
 * region out as an upright image with {@link QuadCrop}, and passes it to a {@link TextRecognizer}. Lines come back
 * in reading order, top to bottom and left to right, each with its text, confidence and position.
 *
 * <pre>{@code
 * try (var ocr = OcrPipeline.builder().build()) {      // PP-OCRv6 small detector and recognizer
 *     for (TextLine line : ocr.read(Path.of("receipt.jpg"))) {
 *         System.out.println(line.text());
 *     }
 * }
 * }</pre>
 *
 * <p>Either stage can be swapped: {@link CraftTextDetector} also fits, although it finds words rather than whole
 * lines, so each word is read separately. Closing the pipeline closes both stages.
 */
public class OcrPipeline implements InferenceTask<BufferedImage, List<TextLine>> {

    private final TextDetector detector;
    private final TextRecognizer recognizer;
    private final float minConfidence;

    private OcrPipeline(TextDetector detector, TextRecognizer recognizer, float minConfidence) {
        this.detector = detector;
        this.recognizer = recognizer;
        this.minConfidence = minConfidence;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Finds and reads the text lines in an image.
     *
     * @return the lines in reading order; lines read as empty or below the minimum confidence are left out
     */
    public List<TextLine> read(BufferedImage image) {
        List<TextLine> lines = new ArrayList<>();
        for (TextRegion region : ReadingOrder.sort(detector.detect(image))) {
            RecognizedText recognized = recognizer.recognize(QuadCrop.crop(image, region.polygon()));
            if (!recognized.text().isEmpty() && recognized.confidence() >= minConfidence) {
                lines.add(new TextLine(recognized.text(), recognized.confidence(), region));
            }
        }
        return lines;
    }

    /** Finds and reads the text lines in an image file. */
    public List<TextLine> read(Path imagePath) {
        return read(PaddleImages.read(imagePath));
    }

    @Override
    public List<TextLine> run(BufferedImage image) {
        return read(image);
    }

    @Override
    public void close() {
        try {
            detector.close();
        } finally {
            recognizer.close();
        }
    }

    public static class Builder {
        private TextDetector detector;
        private TextRecognizer recognizer;
        private float minConfidence = 0f;

        /** The detector that finds text regions. Defaults to {@code PaddleTextDetector.small()}. */
        public Builder detector(TextDetector detector) {
            this.detector = detector;
            return this;
        }

        /** The recognizer that reads each region. Defaults to {@code PaddleTextRecognizer.small()}. */
        public Builder recognizer(TextRecognizer recognizer) {
            this.recognizer = recognizer;
            return this;
        }

        /** Lines read with a lower confidence are left out. Defaults to 0, keeping every non-empty line. */
        public Builder minConfidence(float minConfidence) {
            this.minConfidence = minConfidence;
            return this;
        }

        public OcrPipeline build() {
            TextDetector builtDetector = detector != null ? detector : PaddleTextDetector.small().build();
            TextRecognizer builtRecognizer;
            try {
                builtRecognizer = recognizer != null ? recognizer : PaddleTextRecognizer.small().build();
            } catch (RuntimeException e) {
                if (detector == null) {
                    builtDetector.close();
                }
                throw e;
            }
            return new OcrPipeline(builtDetector, builtRecognizer, minConfidence);
        }
    }
}
