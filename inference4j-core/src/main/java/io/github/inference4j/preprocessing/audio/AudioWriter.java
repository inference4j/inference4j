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

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Writes {@link AudioData} as a 16-bit PCM WAV file.
 *
 * <p>Produces a standard RIFF/WAVE file with a 44-byte header, mono channel,
 * and 16-bit signed PCM encoding. Float samples are clamped to {@code [-1.0, 1.0]}
 * and scaled to the 16-bit signed range.
 */
public final class AudioWriter {

    private static final int HEADER_SIZE = 44;
    private static final int BITS_PER_SAMPLE = 16;
    private static final int BYTES_PER_SAMPLE = BITS_PER_SAMPLE / 8;
    private static final int NUM_CHANNELS = 1;
    private static final short PCM_FORMAT = 1;

    private AudioWriter() {
    }

    /**
     * Writes audio data as a 16-bit PCM WAV file.
     *
     * @param audio the audio data to write (mono float32 samples)
     * @param path  the output file path
     * @throws IOException if the file cannot be written
     */
    public static void write(AudioData audio, Path path) throws IOException {
        float[] samples = audio.samples();
        int sampleRate = audio.sampleRate();
        int dataSize = samples.length * BYTES_PER_SAMPLE;
        int fileSize = HEADER_SIZE + dataSize;

        ByteBuffer buffer = ByteBuffer.allocate(fileSize);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        writeHeader(buffer, sampleRate, dataSize);
        writeSamples(buffer, samples);

        try (OutputStream out = Files.newOutputStream(path)) {
            out.write(buffer.array());
        }
    }

    private static void writeHeader(ByteBuffer buffer, int sampleRate, int dataSize) {
        int byteRate = sampleRate * NUM_CHANNELS * BYTES_PER_SAMPLE;
        short blockAlign = (short) (NUM_CHANNELS * BYTES_PER_SAMPLE);

        // RIFF chunk descriptor
        buffer.put((byte) 'R');
        buffer.put((byte) 'I');
        buffer.put((byte) 'F');
        buffer.put((byte) 'F');
        buffer.putInt(HEADER_SIZE - 8 + dataSize);  // ChunkSize = 36 + SubChunk2Size
        buffer.put((byte) 'W');
        buffer.put((byte) 'A');
        buffer.put((byte) 'V');
        buffer.put((byte) 'E');

        // fmt sub-chunk
        buffer.put((byte) 'f');
        buffer.put((byte) 'm');
        buffer.put((byte) 't');
        buffer.put((byte) ' ');
        buffer.putInt(16);                  // SubChunk1Size (16 for PCM)
        buffer.putShort(PCM_FORMAT);        // AudioFormat (1 = PCM)
        buffer.putShort((short) NUM_CHANNELS);
        buffer.putInt(sampleRate);
        buffer.putInt(byteRate);
        buffer.putShort(blockAlign);
        buffer.putShort((short) BITS_PER_SAMPLE);

        // data sub-chunk
        buffer.put((byte) 'd');
        buffer.put((byte) 'a');
        buffer.put((byte) 't');
        buffer.put((byte) 'a');
        buffer.putInt(dataSize);            // SubChunk2Size
    }

    private static void writeSamples(ByteBuffer buffer, float[] samples) {
        for (float sample : samples) {
            float clamped = Math.max(-1.0f, Math.min(1.0f, sample));
            short pcm = (short) (clamped * 32767);
            buffer.putShort(pcm);
        }
    }
}
