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

/**
 * Segments the object a {@link Prompt} points at.
 *
 * <p>Unlike semantic segmentation, which labels every pixel with a fixed set of classes, a
 * promptable segmenter finds whatever object you indicate, with no notion of what it is. The
 * prompt comes from a person (a click in a UI) or from another model (an {@link ObjectDetector}
 * box), and the result is that object's exact outline as a {@link SegmentationMask}.
 *
 * <p>Work is split in two so one image can be prompted many times cheaply:
 * <pre>{@code
 * ImageFeatures features = segmenter.encode(image);              // slow: once per image
 * SegmentationMask a = segmenter.segment(features, Prompt.point(320, 240));  // fast
 * SegmentationMask b = segmenter.segment(features, Prompt.box(box));         // fast
 * }</pre>
 *
 * <p>This does not extend {@link io.github.inference4j.InferenceTask}: a request needs both an
 * image and a prompt, and the encode/segment split is the point of the API.
 *
 * @see Sam2Segmenter
 * @see SegmentationMask
 */
public interface PromptableSegmenter extends AutoCloseable {

    /**
     * Runs the image encoder. Reuse the result for every prompt on the same image.
     *
     * @param image the image to segment
     * @return the encoded image
     */
    ImageFeatures encode(BufferedImage image);

    /**
     * Runs the image encoder on an image file.
     *
     * @param imagePath path to the image
     * @return the encoded image
     */
    ImageFeatures encode(Path imagePath);

    /**
     * Segments the object the prompt points at, returning the model's best mask.
     *
     * @param features an image encoded by this segmenter
     * @param prompt   clicks and/or a box, in the image's pixel coordinates
     * @return the highest-scoring mask, the size of the encoded image
     */
    SegmentationMask segment(ImageFeatures features, Prompt prompt);

    /**
     * Returns every mask the model proposes for the prompt, best first.
     *
     * <p>A single click is ambiguous: a click on a shirt could mean the shirt or the whole person.
     * SAM answers with several masks at different granularities. {@link #segment} picks the
     * highest-scoring one; use this to offer the alternatives instead.
     *
     * @param features an image encoded by this segmenter
     * @param prompt   clicks and/or a box, in the image's pixel coordinates
     * @return the candidate masks, sorted by descending score
     */
    List<SegmentationMask> segmentCandidates(ImageFeatures features, Prompt prompt);

    /**
     * Encodes and segments in one call. Prefer {@link #encode} plus {@link #segment(ImageFeatures, Prompt)}
     * when prompting the same image more than once.
     *
     * @param image  the image to segment
     * @param prompt clicks and/or a box, in the image's pixel coordinates
     * @return the highest-scoring mask, the size of the image
     */
    default SegmentationMask segment(BufferedImage image, Prompt prompt) {
        return segment(encode(image), prompt);
    }

    /**
     * Encodes and segments an image file in one call.
     *
     * @param imagePath path to the image
     * @param prompt    clicks and/or a box, in the image's pixel coordinates
     * @return the highest-scoring mask, the size of the image
     */
    default SegmentationMask segment(Path imagePath, Prompt prompt) {
        return segment(encode(imagePath), prompt);
    }

    @Override
    void close();
}
