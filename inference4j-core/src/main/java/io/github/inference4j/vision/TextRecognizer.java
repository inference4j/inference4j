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
import java.util.ArrayList;
import java.util.List;

/**
 * Reads the text in an image of a single line, such as a crop produced from a {@link TextDetector} region.
 *
 * <p>Recognizers read one line at a time: they resize their input to a fixed height, so a whole page has to be
 * split into lines first. {@link OcrPipeline} does that by combining a detector with a recognizer.
 *
 * @see PaddleTextRecognizer
 * @see OcrPipeline
 */
public interface TextRecognizer extends InferenceTask<BufferedImage, RecognizedText> {

    /**
     * Reads the text in an image of one line.
     *
     * @param lineImage an image containing a single line of text, roughly horizontal
     * @return the text and its confidence
     */
    RecognizedText recognize(BufferedImage lineImage);

    /** Reads several line images, in order. */
    default List<RecognizedText> recognize(List<BufferedImage> lineImages) {
        List<RecognizedText> results = new ArrayList<>(lineImages.size());
        for (BufferedImage lineImage : lineImages) {
            results.add(recognize(lineImage));
        }
        return results;
    }

    @Override
    default RecognizedText run(BufferedImage lineImage) {
        return recognize(lineImage);
    }

    @Override
    void close();
}
