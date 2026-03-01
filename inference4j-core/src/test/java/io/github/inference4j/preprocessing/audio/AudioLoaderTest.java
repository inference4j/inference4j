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

import org.junit.jupiter.api.Test;

import javax.sound.sampled.AudioFormat;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class AudioLoaderTest {

    @Test
    void toMonoFloat32_16bitMono_littleEndian() {
        AudioFormat fmt = new AudioFormat(16000, 16, 1, true, false);

        // Two samples: 0 and max positive (32767)
        byte[] raw = {
                0x00, 0x00,             // sample 0 → 0.0
                (byte) 0xFF, 0x7F       // sample 1 → 32767/32768 ≈ 1.0
        };

        float[] result = io.github.inference4j.preprocessing.audio.AudioLoader.toMonoFloat32(raw, fmt);

        assertThat(result.length).isEqualTo(2);
        assertThat(result[0]).isCloseTo(0.0f, within(1e-5f));
        assertThat(result[1]).isCloseTo(32767f / 32768f, within(1e-5f));
    }

    @Test
    void toMonoFloat32_16bitMono_bigEndian() {
        AudioFormat fmt = new AudioFormat(16000, 16, 1, true, true);

        // Sample: 256 big-endian → 0x01, 0x00
        byte[] raw = {0x01, 0x00};

        float[] result = io.github.inference4j.preprocessing.audio.AudioLoader.toMonoFloat32(raw, fmt);

        assertThat(result.length).isEqualTo(1);
        assertThat(result[0]).isCloseTo(256f / 32768f, within(1e-5f));
    }

    @Test
    void toMonoFloat32_16bitStereo_averagesChannels() {
        AudioFormat fmt = new AudioFormat(16000, 16, 2, true, false);

        // Frame: left=16384 (0x00, 0x40), right=0 (0x00, 0x00)
        // Average: 8192/32768 = 0.25
        byte[] raw = {
                0x00, 0x40,   // left = 16384
                0x00, 0x00    // right = 0
        };

        float[] result = io.github.inference4j.preprocessing.audio.AudioLoader.toMonoFloat32(raw, fmt);

        assertThat(result.length).isEqualTo(1);
        assertThat(result[0]).isCloseTo(0.25f, within(1e-5f));
    }

    @Test
    void toMonoFloat32_8bitMono() {
        AudioFormat fmt = new AudioFormat(16000, 8, 1, false, false);

        // 8-bit unsigned: 128 → 0.0, 255 → ~1.0, 0 → ~-1.0
        byte[] raw = {(byte) 128, (byte) 255, 0};

        float[] result = io.github.inference4j.preprocessing.audio.AudioLoader.toMonoFloat32(raw, fmt);

        assertThat(result.length).isEqualTo(3);
        assertThat(result[0]).isCloseTo(0.0f, within(1e-5f));
        assertThat(result[1]).isCloseTo(127f / 128f, within(1e-5f));
        assertThat(result[2]).isCloseTo(-1.0f, within(1e-5f));
    }

    @Test
    void toMonoFloat32_negativeValues() {
        AudioFormat fmt = new AudioFormat(16000, 16, 1, true, false);

        // -1 in 16-bit little-endian: 0xFF, 0xFF
        byte[] raw = {(byte) 0xFF, (byte) 0xFF};

        float[] result = io.github.inference4j.preprocessing.audio.AudioLoader.toMonoFloat32(raw, fmt);

        assertThat(result.length).isEqualTo(1);
        assertThat(result[0]).isCloseTo(-1f / 32768f, within(1e-5f));
    }
}
