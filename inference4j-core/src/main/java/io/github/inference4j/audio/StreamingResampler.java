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

package io.github.inference4j.audio;

/**
 * Linear-interpolation resampling for audio that arrives in pieces. Unlike resampling each piece on its
 * own, it carries its position across pieces, so the output has no seams and no timing drift: feeding a
 * recording in pieces gives the same samples as {@code AudioProcessor.resample} on the whole recording,
 * apart from the last output sample, which waits for the next piece.
 */
final class StreamingResampler {

    private final double step;

    /** Source position of the next output sample, relative to the start of the pending piece. */
    private double position;

    /** The last sample of the previous piece, at source index -1. */
    private float previous;
    private boolean started;

    StreamingResampler(int fromRate, int toRate) {
        this.step = (double) fromRate / toRate;
    }

    float[] resample(float[] samples) {
        if (samples.length == 0) {
            return samples;
        }
        // Source index -1 is the previous piece's last sample; the first piece starts at 0
        double pos = started ? position : 0;
        int count = 0;
        float[] out = new float[(int) Math.ceil((samples.length - pos) / step) + 1];
        // Interpolating needs the sample after floor(pos); at the very last sample, pos is a whole number
        while (pos <= samples.length - 1) {
            int i0 = (int) Math.floor(pos);
            float s0 = i0 < 0 ? previous : samples[i0];
            float s1 = samples[Math.min(i0 + 1, samples.length - 1)];
            float frac = (float) (pos - i0);
            out[count++] = s0 * (1f - frac) + s1 * frac;
            pos += step;
        }
        position = pos - samples.length;
        previous = samples[samples.length - 1];
        started = true;
        return java.util.Arrays.copyOf(out, count);
    }
}
