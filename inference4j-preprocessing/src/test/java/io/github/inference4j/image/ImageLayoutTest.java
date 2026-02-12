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
