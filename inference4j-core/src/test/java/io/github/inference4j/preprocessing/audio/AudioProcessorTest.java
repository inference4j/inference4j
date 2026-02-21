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

package io.github.inference4j.preprocessing.audio;

import io.github.inference4j.preprocessing.audio.AudioData;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class AudioProcessorTest {

    @Test
    void resample_identityWhenSameRate() {
        float[] samples = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] result = io.github.inference4j.preprocessing.audio.AudioProcessor.resample(samples, 16000, 16000);

        assertSame(samples, result);
    }

    @Test
    void resample_upsampleDoublesLength() {
        float[] samples = {0.0f, 1.0f};
        float[] result = io.github.inference4j.preprocessing.audio.AudioProcessor.resample(samples, 8000, 16000);

        assertEquals(4, result.length);
        assertEquals(0.0f, result[0], 1e-5f);
        assertEquals(1.0f, result[result.length - 1], 1e-2f);
    }

    @Test
    void resample_downsampleHalvesLength() {
        float[] samples = {0.0f, 0.25f, 0.5f, 0.75f};
        float[] result = io.github.inference4j.preprocessing.audio.AudioProcessor.resample(samples, 16000, 8000);

        assertEquals(2, result.length);
        assertEquals(0.0f, result[0], 1e-5f);
    }

    @Test
    void resample_interpolatesValues() {
        // 4 samples at 4kHz → 2 samples at 2kHz
        float[] samples = {0.0f, 1.0f, 0.0f, 1.0f};
        float[] result = io.github.inference4j.preprocessing.audio.AudioProcessor.resample(samples, 4000, 2000);

        assertEquals(2, result.length);
        assertEquals(0.0f, result[0], 1e-5f);
    }

    @Test
    void normalize_zeroMeanUnitVariance() {
        float[] samples = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] result = io.github.inference4j.preprocessing.audio.AudioProcessor.normalize(samples);

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
        float[] result = io.github.inference4j.preprocessing.audio.AudioProcessor.normalize(samples);

        for (float v : result) {
            assertEquals(0.0f, v, 1e-6f);
        }
    }

    @Test
    void normalize_emptyArray_returnsEmpty() {
        float[] result = io.github.inference4j.preprocessing.audio.AudioProcessor.normalize(new float[0]);
        assertEquals(0, result.length);
    }

    @Test
    void normalize_singleValue_returnsZero() {
        // Single value → std=0, should return zero
        float[] result = io.github.inference4j.preprocessing.audio.AudioProcessor.normalize(new float[]{42.0f});
        assertEquals(0.0f, result[0], 1e-6f);
    }

    // --- chunk() tests ---

    @Test
    void chunk_shorterThanDuration_returnsSingleChunkPaddedToFull() {
        // 3 samples at 3Hz, chunk to 2s → 1 chunk of 6 samples, last 3 zero-padded
        float[] samples = {0.1f, 0.2f, 0.3f};
        AudioData audio = new AudioData(samples, 3);

        List<AudioData> chunks = io.github.inference4j.preprocessing.audio.AudioProcessor.chunk(audio, 2);

        assertEquals(1, chunks.size());
        assertEquals(6, chunks.get(0).samples().length);
        assertEquals(0.1f, chunks.get(0).samples()[0], 1e-6f);
        assertEquals(0.2f, chunks.get(0).samples()[1], 1e-6f);
        assertEquals(0.3f, chunks.get(0).samples()[2], 1e-6f);
        assertEquals(0.0f, chunks.get(0).samples()[3], 1e-6f);
        assertEquals(0.0f, chunks.get(0).samples()[4], 1e-6f);
        assertEquals(0.0f, chunks.get(0).samples()[5], 1e-6f);
    }

    @Test
    void chunk_exactlyDuration_returnsSingleChunk() {
        // 4 samples at 2Hz, chunk to 2s → 1 chunk of 4 samples
        float[] samples = {0.1f, 0.2f, 0.3f, 0.4f};
        AudioData audio = new AudioData(samples, 2);

        List<AudioData> chunks = io.github.inference4j.preprocessing.audio.AudioProcessor.chunk(audio, 2);

        assertEquals(1, chunks.size());
        assertEquals(4, chunks.get(0).samples().length);
        assertArrayEquals(samples, chunks.get(0).samples(), 1e-6f);
    }

    @Test
    void chunk_longerThanDuration_returnsMultipleChunks() {
        // 100 samples at 10Hz (10s), chunk to 3s → 4 chunks of 30 samples each
        float[] samples = new float[100];
        for (int i = 0; i < 100; i++) {
            samples[i] = i * 0.01f;
        }
        AudioData audio = new AudioData(samples, 10);

        List<AudioData> chunks = io.github.inference4j.preprocessing.audio.AudioProcessor.chunk(audio, 3);

        assertEquals(4, chunks.size());
        for (AudioData chunk : chunks) {
            assertEquals(30, chunk.samples().length);
        }
        // First sample of first chunk
        assertEquals(0.0f, chunks.get(0).samples()[0], 1e-6f);
        // First sample of second chunk
        assertEquals(0.30f, chunks.get(1).samples()[0], 1e-6f);
        // Last chunk should have 10 real samples + 20 zero-padded
        assertEquals(0.90f, chunks.get(3).samples()[0], 1e-6f);
        assertEquals(0.0f, chunks.get(3).samples()[10], 1e-6f);
    }

    @Test
    void chunk_allChunksPreserveSampleRate() {
        float[] samples = new float[100];
        AudioData audio = new AudioData(samples, 16000);

        List<AudioData> chunks = io.github.inference4j.preprocessing.audio.AudioProcessor.chunk(audio, 30);

        for (AudioData chunk : chunks) {
            assertEquals(16000, chunk.sampleRate());
        }
    }

    @Test
    void chunk_emptySamples_returnsSingleEmptyPaddedChunk() {
        // Empty audio at 16kHz, chunk to 30s → 1 chunk of 480000 samples
        AudioData audio = new AudioData(new float[0], 16000);

        List<AudioData> chunks = io.github.inference4j.preprocessing.audio.AudioProcessor.chunk(audio, 30);

        assertEquals(1, chunks.size());
        assertEquals(480000, chunks.get(0).samples().length);
        for (float s : chunks.get(0).samples()) {
            assertEquals(0.0f, s, 1e-6f);
        }
    }
}
