package io.github.inference4j.audio;

/**
 * Raw audio waveform data as mono float32 samples.
 *
 * @param samples    PCM samples in {@code [-1.0, 1.0]} range
 * @param sampleRate sample rate in Hz (e.g., 16000, 44100)
 */
public record AudioData(float[] samples, int sampleRate) {
}
