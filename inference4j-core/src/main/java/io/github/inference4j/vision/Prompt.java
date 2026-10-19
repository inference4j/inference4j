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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Tells a {@link PromptableSegmenter} which object to segment.
 *
 * <p>A prompt is any mix of clicks and at most one box, all in pixel coordinates of the
 * original image. A foreground click says "this pixel belongs to the object"; a background
 * click says "this pixel does not", which is how a user trims a mask that took too much.
 * A box says "the object is inside here", which is what an {@link ObjectDetector} gives you.
 *
 * <p>Prompts are immutable: each {@code with...} method returns a new prompt.
 *
 * <pre>{@code
 * Prompt.point(320, 240);                              // one click
 * Prompt.point(320, 240).withBackground(300, 400);     // click, then exclude a region
 * Prompt.box(detection.box());                         // a detector's box
 * Prompt.box(detection.box()).withPoint(320, 240);     // box plus a hint inside it
 * }</pre>
 *
 * @see PromptableSegmenter
 */
public final class Prompt {

    /**
     * One click on the image.
     *
     * @param point      where the click landed, in original image pixels
     * @param foreground {@code true} if the pixel belongs to the object, {@code false} if it does not
     */
    public record Click(Point point, boolean foreground) {
    }

    private final List<Click> clicks;
    private final BoundingBox box;

    private Prompt(List<Click> clicks, BoundingBox box) {
        this.clicks = List.copyOf(clicks);
        this.box = box;
    }

    /**
     * A single foreground click.
     *
     * @param x column, from the left edge
     * @param y row, from the top edge
     * @return a new prompt
     */
    public static Prompt point(float x, float y) {
        return point(new Point(x, y));
    }

    /**
     * A single foreground click.
     *
     * @param point the clicked pixel
     * @return a new prompt
     */
    public static Prompt point(Point point) {
        return new Prompt(List.of(new Click(point, true)), null);
    }

    /**
     * A box around the object.
     *
     * @param x1 left edge
     * @param y1 top edge
     * @param x2 right edge
     * @param y2 bottom edge
     * @return a new prompt
     */
    public static Prompt box(float x1, float y1, float x2, float y2) {
        return box(new BoundingBox(x1, y1, x2, y2));
    }

    /**
     * A box around the object, such as {@link Detection#box()} from an {@link ObjectDetector}.
     *
     * @param box the box, in original image pixels
     * @return a new prompt
     */
    public static Prompt box(BoundingBox box) {
        return new Prompt(List.of(), box);
    }

    /**
     * Adds a foreground click.
     *
     * @param x column, from the left edge
     * @param y row, from the top edge
     * @return a new prompt with the click added
     */
    public Prompt withPoint(float x, float y) {
        return withClick(new Click(new Point(x, y), true));
    }

    /**
     * Adds a background click: a pixel that must not be part of the mask.
     *
     * @param x column, from the left edge
     * @param y row, from the top edge
     * @return a new prompt with the click added
     */
    public Prompt withBackground(float x, float y) {
        return withClick(new Click(new Point(x, y), false));
    }

    /**
     * Sets the box, replacing any box this prompt already has.
     *
     * @param box the box, in original image pixels
     * @return a new prompt with the box set
     */
    public Prompt withBox(BoundingBox box) {
        return new Prompt(clicks, box);
    }

    /** The clicks, in the order they were added. */
    public List<Click> clicks() {
        return clicks;
    }

    /** The box, if this prompt has one. */
    public Optional<BoundingBox> box() {
        return Optional.ofNullable(box);
    }

    private Prompt withClick(Click click) {
        List<Click> added = new ArrayList<>(clicks);
        added.add(click);
        return new Prompt(added, box);
    }

    @Override
    public String toString() {
        return "Prompt[clicks=" + clicks + ", box=" + box + "]";
    }
}
