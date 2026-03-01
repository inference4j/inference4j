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
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

class AudioWriterTest {

    @TempDir
    Path tempDir;

    @Test
    void writeAndReadBack_preservesSamples() throws IOException {
        float[] samples = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
        AudioData audio = new AudioData(samples, 16000);
        Path file = tempDir.resolve("test.wav");

        io.github.inference4j.preprocessing.audio.AudioWriter.write(audio, file);
        AudioData loaded = io.github.inference4j.preprocessing.audio.AudioLoader.load(file);

        assertThat(loaded.samples().length).isEqualTo(samples.length);
        float tolerance = 2.0f / 32768;
        for (int i = 0; i < samples.length; i++) {
            assertThat(loaded.samples()[i]).as("Sample " + i + " mismatch")
                    .isCloseTo(samples[i], within(tolerance));
        }
    }

    @Test
    void writeAndReadBack_emptySamples() throws IOException {
        float[] samples = {};
        AudioData audio = new AudioData(samples, 16000);
        Path file = tempDir.resolve("empty.wav");

        io.github.inference4j.preprocessing.audio.AudioWriter.write(audio, file);
        AudioData loaded = io.github.inference4j.preprocessing.audio.AudioLoader.load(file);

        assertThat(loaded.samples().length).isEqualTo(0);
    }

    @Test
    void writeAndReadBack_differentSampleRate() throws IOException {
        float[] samples = {0.0f, 0.25f, -0.25f};
        AudioData audio = new AudioData(samples, 44100);
        Path file = tempDir.resolve("rate.wav");

        io.github.inference4j.preprocessing.audio.AudioWriter.write(audio, file);
        AudioData loaded = io.github.inference4j.preprocessing.audio.AudioLoader.load(file);

        assertThat(loaded.sampleRate()).isEqualTo(44100);
    }

    @Test
    void write_createsValidWavFile() throws IOException {
        float[] samples = {0.1f, 0.2f, 0.3f};
        AudioData audio = new AudioData(samples, 16000);
        Path file = tempDir.resolve("size.wav");

        io.github.inference4j.preprocessing.audio.AudioWriter.write(audio, file);

        // 44-byte header + 3 samples * 2 bytes each = 50 bytes
        assertThat(Files.size(file)).isEqualTo(50);
    }
}
