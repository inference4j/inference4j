package io.github.inference4j.audio;

/**
 * Audio signal processing utilities for model inference.
 */
public final class AudioProcessor {

    private AudioProcessor() {
    }

    /**
     * Resamples audio from one sample rate to another using linear interpolation.
     *
     * <p>Returns the input unchanged if {@code fromRate == toRate}.
     *
     * @param samples  input samples
     * @param fromRate source sample rate in Hz
     * @param toRate   target sample rate in Hz
     * @return resampled audio
     */
    public static float[] resample(float[] samples, int fromRate, int toRate) {
        if (fromRate == toRate) {
            return samples;
        }

        double ratio = (double) fromRate / toRate;
        int outputLength = (int) Math.ceil(samples.length / ratio);
        float[] output = new float[outputLength];

        for (int i = 0; i < outputLength; i++) {
            double srcIndex = i * ratio;
            int idx0 = (int) srcIndex;
            int idx1 = Math.min(idx0 + 1, samples.length - 1);
            float frac = (float) (srcIndex - idx0);
            output[i] = samples[idx0] * (1f - frac) + samples[idx1] * frac;
        }

        return output;
    }

    /**
     * Normalizes audio to zero mean and unit variance.
     *
     * <p>Computes {@code (x - mean) / std} for each sample. If the standard
     * deviation is zero (silent audio), returns a zero-filled array.
     *
     * @param samples input samples
     * @return normalized samples
     */
    public static float[] normalize(float[] samples) {
        if (samples.length == 0) {
            return new float[0];
        }

        double sum = 0.0;
        for (float s : samples) {
            sum += s;
        }
        float mean = (float) (sum / samples.length);

        double sumSq = 0.0;
        for (float s : samples) {
            double diff = s - mean;
            sumSq += diff * diff;
        }
        float std = (float) Math.sqrt(sumSq / samples.length);

        float[] result = new float[samples.length];
        if (std == 0f) {
            return result;
        }

        for (int i = 0; i < samples.length; i++) {
            result[i] = (samples[i] - mean) / std;
        }

        return result;
    }
}
