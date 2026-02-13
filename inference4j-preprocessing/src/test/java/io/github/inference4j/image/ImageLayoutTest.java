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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ImageLayoutTest {

    @Test
    void detect_nchw_rgb() {
        ImageLayout layout = ImageLayout.detect(new long[]{1, 3, 224, 224});
        assertEquals(ImageLayout.NCHW, layout);
    }

    @Test
    void detect_nhwc_rgb() {
        ImageLayout layout = ImageLayout.detect(new long[]{1, 224, 224, 3});
        assertEquals(ImageLayout.NHWC, layout);
    }

    @Test
    void detect_nchw_singleChannel() {
        ImageLayout layout = ImageLayout.detect(new long[]{1, 1, 28, 28});
        assertEquals(ImageLayout.NCHW, layout);
    }

    @Test
    void detect_nhwc_singleChannel() {
        ImageLayout layout = ImageLayout.detect(new long[]{1, 28, 28, 1});
        assertEquals(ImageLayout.NHWC, layout);
    }

    @Test
    void detect_ambiguous_defaultsToNCHW() {
        // [1, 3, 3, 3] — channels could be dim 1 or dim 3
        ImageLayout layout = ImageLayout.detect(new long[]{1, 3, 3, 3});
        assertEquals(ImageLayout.NCHW, layout);
    }

    @Test
    void detect_nonFourDimensional_throws() {
        assertThrows(IllegalArgumentException.class,
                () -> ImageLayout.detect(new long[]{1, 3, 224}));
    }

    @Test
    void imageSize_nchw() {
        assertEquals(224, ImageLayout.NCHW.imageSize(new long[]{1, 3, 224, 224}));
    }

    @Test
    void imageSize_nhwc() {
        assertEquals(224, ImageLayout.NHWC.imageSize(new long[]{1, 224, 224, 3}));
    }
}
