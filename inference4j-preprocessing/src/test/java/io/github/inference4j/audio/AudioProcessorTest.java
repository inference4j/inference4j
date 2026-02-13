package io.github.inference4j.audio;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AudioProcessorTest {

    @Test
    void resample_identityWhenSameRate() {
        float[] samples = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] result = AudioProcessor.resample(samples, 16000, 16000);

        assertSame(samples, result);
    }

    @Test
    void resample_upsampleDoublesLength() {
        float[] samples = {0.0f, 1.0f};
        float[] result = AudioProcessor.resample(samples, 8000, 16000);

        assertEquals(4, result.length);
        assertEquals(0.0f, result[0], 1e-5f);
        assertEquals(1.0f, result[result.length - 1], 1e-2f);
    }

    @Test
    void resample_downsampleHalvesLength() {
        float[] samples = {0.0f, 0.25f, 0.5f, 0.75f};
        float[] result = AudioProcessor.resample(samples, 16000, 8000);

        assertEquals(2, result.length);
        assertEquals(0.0f, result[0], 1e-5f);
    }

    @Test
    void resample_interpolatesValues() {
        // 4 samples at 4kHz → 2 samples at 2kHz
        float[] samples = {0.0f, 1.0f, 0.0f, 1.0f};
        float[] result = AudioProcessor.resample(samples, 4000, 2000);

        assertEquals(2, result.length);
        assertEquals(0.0f, result[0], 1e-5f);
    }

    @Test
    void normalize_zeroMeanUnitVariance() {
        float[] samples = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] result = AudioProcessor.normalize(samples);

        assertEquals(samples.length, result.length);

        // Check mean ≈ 0
        double mean = 0;
        for (float v : result) mean += v;
        mean /= result.length;
        assertEquals(0.0, mean, 1e-5);

        // Check std ≈ 1
        double sumSq = 0;
        for (float v : result) sumSq += (v - mean) * (v - mean);
        double std = Math.sqrt(sumSq / result.length);
        assertEquals(1.0, std, 1e-5);
    }

    @Test
    void normalize_silentAudio_returnsZeros() {
        float[] samples = {0.0f, 0.0f, 0.0f};
        float[] result = AudioProcessor.normalize(samples);

        for (float v : result) {
            assertEquals(0.0f, v, 1e-6f);
        }
    }

    @Test
    void normalize_emptyArray_returnsEmpty() {
        float[] result = AudioProcessor.normalize(new float[0]);
        assertEquals(0, result.length);
    }

    @Test
    void normalize_singleValue_returnsZero() {
        // Single value → std=0, should return zero
        float[] result = AudioProcessor.normalize(new float[]{42.0f});
        assertEquals(0.0f, result[0], 1e-6f);
    }
}
