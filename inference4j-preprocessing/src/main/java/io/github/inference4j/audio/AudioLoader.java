package io.github.inference4j.audio;

import io.github.inference4j.exception.InferenceException;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Loads audio files into {@link AudioData} for model inference.
 *
 * <p>Supports WAV files via {@code javax.sound.sampled}. Automatically converts
 * to mono float32 regardless of the source format (8/16-bit, mono/stereo,
 * big/little endian).
 */
public final class AudioLoader {

    private AudioLoader() {
    }

    /**
     * Loads a WAV file and converts it to mono float32 samples.
     *
     * @param path path to the WAV file
     * @return audio data with mono float32 samples and the file's sample rate
     * @throws InferenceException if the file cannot be read or has an unsupported format
     */
    public static AudioData load(Path path) {
        try (AudioInputStream ais = AudioSystem.getAudioInputStream(path.toFile())) {
            AudioFormat fmt = ais.getFormat();
            byte[] raw = ais.readAllBytes();
            float[] samples = toMonoFloat32(raw, fmt);
            return new AudioData(samples, (int) fmt.getSampleRate());
        } catch (UnsupportedAudioFileException e) {
            throw new InferenceException("Unsupported audio format: " + path, e);
        } catch (IOException e) {
            throw new InferenceException("Failed to read audio file: " + path, e);
        }
    }

    /**
     * Converts raw PCM bytes to mono float32 samples in {@code [-1.0, 1.0]}.
     *
     * <p>Handles 8-bit unsigned and 16-bit signed PCM, mono and stereo,
     * big and little endian.
     */
    static float[] toMonoFloat32(byte[] raw, AudioFormat fmt) {
        int channels = fmt.getChannels();
        int sampleSizeInBits = fmt.getSampleSizeInBits();
        boolean bigEndian = fmt.isBigEndian();
        int bytesPerSample = sampleSizeInBits / 8;
        int frameSize = bytesPerSample * channels;
        int numFrames = raw.length / frameSize;

        float[] mono = new float[numFrames];

        for (int i = 0; i < numFrames; i++) {
            float sum = 0f;
            for (int ch = 0; ch < channels; ch++) {
                int offset = i * frameSize + ch * bytesPerSample;
                float sample = decodeSample(raw, offset, sampleSizeInBits, bigEndian);
                sum += sample;
            }
            mono[i] = sum / channels;
        }

        return mono;
    }

    private static float decodeSample(byte[] raw, int offset, int bits, boolean bigEndian) {
        if (bits == 8) {
            // 8-bit PCM is unsigned: 0-255, center at 128
            int unsigned = raw[offset] & 0xFF;
            return (unsigned - 128) / 128f;
        } else if (bits == 16) {
            int lo, hi;
            if (bigEndian) {
                hi = raw[offset];
                lo = raw[offset + 1] & 0xFF;
            } else {
                lo = raw[offset] & 0xFF;
                hi = raw[offset + 1];
            }
            short sample = (short) ((hi << 8) | lo);
            return sample / 32768f;
        } else {
            throw new InferenceException("Unsupported sample size: " + bits + " bits");
        }
    }
}
