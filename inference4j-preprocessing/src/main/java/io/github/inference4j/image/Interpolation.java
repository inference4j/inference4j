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

package io.github.inference4j.image;

import java.awt.RenderingHints;

/**
 * Interpolation method used when resizing images.
 *
 * @see ResizeTransform
 */
public enum Interpolation {

    BILINEAR(RenderingHints.VALUE_INTERPOLATION_BILINEAR),
    BICUBIC(RenderingHints.VALUE_INTERPOLATION_BICUBIC),
    NEAREST_NEIGHBOR(RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);

    private final Object hint;

    Interpolation(Object hint) {
        this.hint = hint;
    }

    Object renderingHint() {
        return hint;
    }
}
